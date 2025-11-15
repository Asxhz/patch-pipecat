#
# Copyright (c) 2024–2025, Daily
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Pipecat Transcription-Only → RWKV (Gradio)

- Streams mic audio (Daily/WebRTC) to Deepgram STT
- Prints raw transcript lines live to stdout
- Debounces chunks and sends combined text to RWKV Gradio (BlinkDL/RWKV-Gradio-1, /evaluate)
- Prints enhanced output from RWKV

Run:
    uv run bot.py --transport daily
    # or
    uv run bot.py --transport webrtc

Env:
    DEEPGRAM_API_KEY=...
    # Optional overrides for RWKV call (defaults match your example):
    RMKV_SPACE_ID="BlinkDL/RWKV-Gradio-1"
    RMKV_API_NAME="/evaluate"
    RMKV_DEBOUNCE_SECS=1.2
    RMKV_TOKEN_COUNT=1000
    RMKV_TEMPERATURE=1.0
    RMKV_TOP_P=0.3
    RMKV_PRESENCE_PENALTY=0.5
    RMKV_COUNT_PENALTY=0.5
"""

import os
import asyncio
from typing import Optional
import math

from dotenv import load_dotenv
from loguru import logger

print("🎙️ Pipecat → Deepgram → RWKV (Gradio)")
load_dotenv(override=True)

# --- Turn-taking & VAD are configured on the transport (NOT inside the pipeline)
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# --- Pipecat core
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams

# --- STT + transcript capture
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.processors.transcript_processor import TranscriptProcessor

# --- Gradio client (RWKV)
from gradio_client import Client


class RWKVDebouncedSender:
    """
    Buffers transcript text and calls RWKV Gradio after a short pause (debounce).
    Defaults match your example call to BlinkDL/RWKV-Gradio-1 /evaluate.
    """
    def __init__(
        self,
        wait_secs: float = 1.2,
        *,
        space_id: str = "BlinkDL/RWKV-Gradio-1",
        api_name: str = "/evaluate",
        token_count: int = 1000,
        temperature: float = 1.0,
        top_p: float = 0.3,
        presence_penalty: float = 0.5,
        count_penalty: float = 0.5,
    ):
        self.client = Client(space_id)
        self.api_name = api_name
        self.wait_secs = wait_secs
        self.token_count = token_count
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.count_penalty = count_penalty

        self._buffer: list[str] = []
        self._timer_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def add_text(self, text: str):
        self._buffer.append(text)
        self._restart_timer()

    def _restart_timer(self):
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        self._timer_task = asyncio.create_task(self._debounced_fire())

    async def _debounced_fire(self):
        try:
            await asyncio.sleep(self.wait_secs)
            await self._send_now()
        except asyncio.CancelledError:
            pass  # restarted; ignore

    def _build_ctx(self, user_text: str) -> str:
        # Matches your example prompt format exactly:
        # ctx="User: <...>\n\nAssistant: <think"
        return f"User: {user_text}\n\nAssistant: <think"

    async def _send_now(self):
        async with self._lock:
            if not self._buffer:
                return
            payload = " ".join(self._buffer).strip()
            self._buffer.clear()

        if not payload:
            return

        try:
            ctx = self._build_ctx(payload)
            result = self.client.predict(
                ctx=ctx,
                token_count=self.token_count,
                temperature=self.temperature,
                top_p=self.top_p,
                presencePenalty=self.presence_penalty,
                countPenalty=self.count_penalty,
                api_name=self.api_name,
            )

            print("\n——— RWKV (Enhanced) ———")
            if isinstance(result, (list, tuple)):
                for item in result:
                    print(str(item))
            else:
                print(str(result))
            print("——— End RWKV ——\n")
        except Exception as e:
            logger.exception(f"RWKV Gradio call failed: {e}")

    async def flush_and_close(self):
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        await self._send_now()


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    if not os.getenv("DEEPGRAM_API_KEY"):
        raise RuntimeError("DEEPGRAM_API_KEY is not set.")

    # RWKV config (env overrides optional)
    debouncer = RWKVDebouncedSender(
        wait_secs=float(os.getenv("RMKV_DEBOUNCE_SECS", "1.2")),
        space_id=os.getenv("RMKV_SPACE_ID", "BlinkDL/RWKV-Gradio-1"),
        api_name=os.getenv("RMKV_API_NAME", "/evaluate"),
        token_count=int(os.getenv("RMKV_TOKEN_COUNT", "1000")),
        temperature=float(os.getenv("RMKV_TEMPERATURE", "1.0")),
        top_p=float(os.getenv("RMKV_TOP_P", "0.3")),
        presence_penalty=float(os.getenv("RMKV_PRESENCE_PENALTY", "0.5")),
        count_penalty=float(os.getenv("RMKV_COUNT_PENALTY", "0.5")),
    )

    # Services/processors
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    transcript = TranscriptProcessor()
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Live transcript + queue for RWKV
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(proc, frame):
        for m in frame.messages:
            if getattr(m, "role", None) == "user" and (txt := getattr(m, "content", "")):
                print(txt, flush=True)     # raw transcript to stdout
                debouncer.add_text(txt)    # debounce to RWKV

    # IMPORTANT: VAD/turn analyzers are NOT pipeline processors.
    # They are already attached to the transport (see bot()).
    pipeline = Pipeline(
        [
            transport.input(),   # mic audio frames
            rtvi,                # optional framework processor
            stt,                 # Deepgram STT
            transcript.user(),   # capture transcripts
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=False, enable_usage_metrics=False),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_t, _c):
        logger.info("Client connected — start speaking.")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_t, _c):
        logger.info("Client disconnected — flushing RWKV.")
        await debouncer.flush_and_close()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    # Attach VAD & turn analyzers to the transport (correct place).
    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=False,  # no TTS
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=False,  # no TTS
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
