package com.hyngng.justagem.ui.voice

import android.app.Application
import com.hyngng.justagem.audio.AppFeedbackCue
import com.hyngng.justagem.audio.SpeechEvent
import com.hyngng.justagem.audio.SpeechInput
import com.hyngng.justagem.audio.SpeechOutput
import com.hyngng.justagem.ui.ChatUiState
import com.hyngng.justagem.ui.VoicePhase
import com.hyngng.justagem.ui.VoiceUi
import com.hyngng.justagem.ui.i18n.AppLanguage
import com.hyngng.justagem.ui.i18n.tr
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Owns everything behind the voice overlay: STT session, the sentence-sized
 * TTS streaming queue, replay caption state and the auto-listen heartbeat.
 * It mutates only the `voice` slice of `ChatUiState` through [mutate], so the
 * ViewModel stays the single owner of conversation state.
 */
class VoiceSessionController(
    application: Application,
    private val scope: kotlinx.coroutines.CoroutineScope,
    private val state: () -> ChatUiState,
    private val mutate: ((ChatUiState) -> ChatUiState) -> Unit,
    private val isGenerating: () -> Boolean,
    /** Starts a full turn from a recognized utterance (append, feedback, generate). */
    private val beginVoiceTurn: (utterance: String) -> Unit,
    private val stopGenerationNow: () -> Unit,
) {

    private val speechIn = SpeechInput(application)
    private val speechOut = SpeechOutput(application)

    private var listenJob: Job? = null
    private var autoListenJob: Job? = null
    private var inputPulseJob: Job? = null
    private var inputPulseVersion = 0L
    private var speechQueueJob: Job? = null
    private var speechChannel: Channel<String>? = null

    fun enterVoiceMode() {
        // Do not grab the microphone before the user asks: the overlay opens
        // in a clear "tap to talk" state, so a quick tap can never cut a
        // listening session short by accident.
        mutate { it.copy(voice = VoiceUi(active = true, phase = VoicePhase.IDLE)) }
    }

    fun exitVoiceMode() {
        stopVoice(closeOverlay = true)
    }

    fun onVoicePrimaryAction() {
        when (state().voice.phase) {
            VoicePhase.IDLE -> {
                startListening()
            }
            VoicePhase.LISTENING -> {
                speechIn.stop()
                // Some recognizers never answer after stopListening(); don't
                // leave the orb stuck in LISTENING.
                autoListenJob?.cancel()
                autoListenJob = scope.launch {
                    delay(2_000)
                    if (state().voice.active && state().voice.phase == VoicePhase.LISTENING) {
                        listenJob?.cancel()
                        listenJob = null
                        speechIn.stop()
                        mutate { it.copy(
                            voice = it.voice.copy(
                                phase = VoicePhase.IDLE,
                                hint = tr("듣지 못했어요. 다시 말씀해 주세요.", "I didn't catch that. Please say it again."),
                            ),
                        ) }
                    }
                }
            }

            VoicePhase.THINKING -> {
                interruptAndListen()
            }

            VoicePhase.SPEAKING -> {
                interruptAndListen()
            }
        }
    }

    /**
     * Immediately halts any ongoing assistant generation and TTS playback,
     * transitioning the session directly to listening to the user's voice.
     */
    fun interruptAndListen() {
        speechQueueJob?.cancel()
        speechQueueJob = null
        speechChannel?.cancel()
        speechChannel = null
        speechOut.stop()
        if (isGenerating()) {
            stopGenerationNow()
        }
        startListening()
    }

    fun startListening() {
        if (!state().voice.active) return
        if (isGenerating()) return
        autoListenJob?.cancel()
        listenJob?.cancel()
        clearVoiceInputLevel()
        speechOut.stop()
        mutate { it.copy(
            voice = it.voice.copy(
                phase = VoicePhase.LISTENING,
                transcript = "",
                caption = "",
                hint = "",
            ),
        ) }
        listenJob = scope.launch {
            var terminal = false
            var hasInputLevel = false
            try {
                val speechLocale = AppLanguage.resolve(state().languageTag).locale
                speechIn.listen(speechLocale).collect { event ->
                    val voice = state().voice
                    if (!voice.active || voice.phase != VoicePhase.LISTENING) {
                        return@collect
                    }
                    when (event) {
                        is SpeechEvent.InputStarted -> {
                            if (!hasInputLevel) pulseVoiceInput(0.48f)
                        }

                        is SpeechEvent.InputLevel -> {
                            hasInputLevel = true
                            setVoiceInputLevel(event.level)
                        }

                        is SpeechEvent.Partial -> {
                            if (event.text != voice.transcript) {
                                mutate { it.copy(voice = it.voice.copy(transcript = event.text)) }
                                if (!hasInputLevel) pulseVoiceInput(0.55f)
                            }
                        }

                        is SpeechEvent.Final -> {
                            terminal = true
                            val text = event.text.trim()
                            if (text.isBlank()) {
                                mutate { it.copy(
                                    voice = it.voice.copy(
                                        phase = VoicePhase.IDLE,
                                        transcript = "",
                                        caption = "",
                                        inputLevel = 0f,
                                        hint = tr("듣지 못했어요. 다시 말씀해 주세요.", "I didn't catch that. Please say it again."),
                                    ),
                                ) }
                            } else {
                                mutate { it.copy(
                                    voice = it.voice.copy(
                                        phase = VoicePhase.THINKING,
                                        transcript = text,
                                        caption = text,
                                        inputLevel = 0f,
                                        hint = "",
                                    ),
                                ) }
                                beginVoiceTurn(text)
                            }
                        }

                        is SpeechEvent.Failed -> {
                            terminal = true
                            mutate { it.copy(
                                voice = it.voice.copy(
                                    phase = VoicePhase.IDLE,
                                    inputLevel = 0f,
                                    hint = event.message,
                                ),
                            ) }
                        }
                    }
                }
            } catch (c: CancellationException) {
                throw c
            } catch (t: Throwable) {
                terminal = true
                if (state().voice.active) {
                    mutate { it.copy(
                        voice = it.voice.copy(
                            phase = VoicePhase.IDLE,
                            inputLevel = 0f,
                            hint = tr("음성 인식을 사용할 수 없어요.", "Speech recognition is unavailable."),
                        ),
                    ) }
                }
            } finally {
                // The flow ended without a final/error event (for example the
                // engine was torn down). Never leave the UI claiming to listen.
                if (!terminal && state().voice.active &&
                    state().voice.phase == VoicePhase.LISTENING
                ) {
                    mutate { it.copy(
                        voice = it.voice.copy(
                            phase = VoicePhase.IDLE,
                            inputLevel = 0f,
                            hint = tr("다시 탭하여 말씀해 주세요.", "Tap again to speak."),
                        ),
                    ) }
                }
            }
        }
    }

    /**
     * Streams an in-progress answer turn to TTS in sentence-sized pieces and
     * mirrors the reply into the overlay caption. One instance per voice turn.
     */
    class VoiceReplySpeaker internal constructor(
        internal val channel: Channel<String>,
        internal val queueJob: Job,
        private val onCaption: (String) -> Unit,
        private val splitBoundary: (String) -> Int,
    ) {
        private var buffer = ""

        suspend fun onAnswerText(fullText: String, delta: String) {
            onCaption(fullText)
            buffer += delta
            var boundary = splitBoundary(buffer)
            while (boundary != -1) {
                channel.send(buffer.take(boundary))
                buffer = buffer.drop(boundary).trimStart()
                boundary = splitBoundary(buffer)
            }
        }

        suspend fun completeInput() {
            if (buffer.isNotBlank()) channel.send(buffer)
            channel.close()
        }

        /** Answer stream aborted (cancel/error): stop speaking the rest. */
        fun abort() {
            channel.close()
        }
    }

    /** Creates the per-turn TTS queue. `fromVoice == false` keeps a text turn speechless. */
    fun beginReply(fromVoice: Boolean): VoiceReplySpeaker? {
        if (!fromVoice) return null
        val channel = Channel<String>(Channel.UNLIMITED)
        val job = scope.launch { speakReplyChunks(channel) }
        speechChannel = channel
        speechQueueJob = job
        return VoiceReplySpeaker(
            channel = channel,
            queueJob = job,
            onCaption = { caption -> mutate { it.copy(voice = it.voice.copy(caption = caption)) } },
            splitBoundary = ::findSpeechBoundary,
        )
    }

    suspend fun joinReplyQueue() {
        speechQueueJob?.join()
    }

    /** Titles kept in a field while the turn ran are harmless after it ends; never clear a newer turn's slot. */
    fun finishReplyTurn(speaker: VoiceReplySpeaker?) {
        speaker ?: return
        if (speechChannel === speaker.channel) speechChannel = null
        if (speechQueueJob === speaker.queueJob && speechQueueJob?.isCompleted == true) {
            speechQueueJob = null
        }
    }

    /**
     * Ends any active voice session. The overlay state resets fully when the
     * overlay itself closes; a remaining visible overlay only drops to idle.
     */
    fun stopVoice(closeOverlay: Boolean) {
        autoListenJob?.cancel()
        autoListenJob = null
        listenJob?.cancel()
        listenJob = null
        speechQueueJob?.cancel()
        speechQueueJob = null
        speechChannel?.cancel()
        speechChannel = null
        speechIn.stop()
        speechOut.stop()
        mutate { it.copy(voice = it.voice.copy(active = !closeOverlay)) }
        if (closeOverlay) {
            mutate { it.copy(voice = VoiceUi()) }
        }
    }

    // ----------------------------------------------------------------- tts

    private suspend fun speakReplyChunks(chunks: Channel<String>) {
        var spokenCharacters = 0
        for (chunk in chunks) {
            if (chunk.isBlank() || !state().voice.active) continue
            if (state().voice.phase != VoicePhase.SPEAKING) {
                mutate { it.copy(
                    voice = it.voice.copy(
                        phase = VoicePhase.SPEAKING,
                        transcript = "",
                        caption = state().voice.caption.ifBlank { chunk },
                        captionProgress = 0f,
                        speechCharacterIndex = 0,
                        speechLevel = 0f,
                        hint = "",
                    ),
                ) }
            }
            val chunkStart = spokenCharacters
            val speechLocale = AppLanguage.resolve(state().languageTag).locale
            speechOut.speak(
                text = chunk,
                locale = speechLocale,
                onProgress = { progress ->
                    if (state().voice.active && state().voice.phase == VoicePhase.SPEAKING) {
                        val total = state().voice.caption.length.coerceAtLeast(1)
                        val absolute = (chunkStart + progress * chunk.length) / total
                        mutate { it.copy(
                            voice = it.voice.copy(
                                captionProgress = absolute.coerceIn(0f, 1f),
                                speechCharacterIndex = (chunkStart + progress * chunk.length).toInt(),
                            ),
                        ) }
                    }
                },
                onAmplitude = { level ->
                    if (state().voice.active && state().voice.phase == VoicePhase.SPEAKING) {
                        mutate { it.copy(voice = it.voice.copy(speechLevel = level.coerceIn(0f, 1f))) }
                    }
                },
            )
            spokenCharacters += chunk.length
        }
        if (state().voice.active && state().voice.phase == VoicePhase.SPEAKING) {
            mutate { it.copy(
                voice = it.voice.copy(
                    captionProgress = 1f,
                    speechCharacterIndex = state().voice.caption.length,
                    speechLevel = 0f,
                ),
            ) }
            scheduleListening()
        }
    }

    private fun findSpeechBoundary(text: String): Int {
        val index = text.indexOfFirst { it in ".!?。！？\n" }
        return if (index >= 0) index + 1 else -1
    }

    private fun scheduleListening(delayMs: Long = 520) {
        autoListenJob?.cancel()
        autoListenJob = scope.launch {
            delay(delayMs)
            if (state().voice.active && !isGenerating()) startListening()
        }
    }

    // ------------------------------------------------------------ metering

    private fun pulseVoiceInput(level: Float) {
        val version = ++inputPulseVersion
        inputPulseJob?.cancel()
        mutate { it.copy(voice = it.voice.copy(inputLevel = level)) }
        inputPulseJob = scope.launch {
            delay(180)
            if (version == inputPulseVersion && state().voice.phase == VoicePhase.LISTENING) {
                mutate { it.copy(voice = it.voice.copy(inputLevel = 0f)) }
            }
        }
    }

    private fun setVoiceInputLevel(level: Float) {
        inputPulseVersion++
        inputPulseJob?.cancel()
        inputPulseJob = null
        mutate { it.copy(voice = it.voice.copy(inputLevel = level.coerceIn(0f, 1f))) }
    }

    private fun clearVoiceInputLevel() {
        inputPulseVersion++
        inputPulseJob?.cancel()
        inputPulseJob = null
        if (state().voice.inputLevel != 0f) {
            mutate { it.copy(voice = it.voice.copy(inputLevel = 0f)) }
        }
    }

    fun close() {
        stopVoice(closeOverlay = true)
        speechIn.close()
        speechOut.close()
    }
}
