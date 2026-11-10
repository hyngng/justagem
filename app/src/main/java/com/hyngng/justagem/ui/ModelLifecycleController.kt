package com.hyngng.justagem.ui

import com.google.mlkit.genai.common.FeatureStatus
import com.hyngng.justagem.ai.DownloadEvent
import com.hyngng.justagem.ai.GemmaEngine
import com.hyngng.justagem.ai.ModelOption
import com.hyngng.justagem.ui.i18n.tr
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

/**
 * Owns the model handle lifecycle: the background probe, the asset download,
 * the quiet accelerator warmup, engine selection, and the selection the user
 * made mid-stream (applied only when the turn owning AICore has ended).
 * It reads `ChatUiState` through [state]/[mutate]; turn gating notices stay in
 * the ViewModel facade.
 */
class ModelLifecycleController(
    private val engine: GemmaEngine,
    private val scope: CoroutineScope,
    private val state: () -> ChatUiState,
    private val mutate: ((ChatUiState) -> ChatUiState) -> Unit,
    private val postNotice: (String) -> Unit,
    /** Called when a probe resolved a base model, so it can be saved. */
    private val onProbeResolvedModel: () -> Unit,
) {

    private var probeJob: Job? = null
    private var downloadJob: Job? = null
    // Pre-loads the accelerator once after the model turns READY (probe or
    // download), or on the first composer touch, so the first real turn
    // skips cold start.
    private var warmupJob: Job? = null
    // Model chosen during streaming; applied when the current turn ends.
    private var pendingModel: ModelOption? = null

    fun probe() {
        probeJob?.cancel()
        mutate { it.copy(model = it.model.copy(phase = ModelPhase.CHECKING, error = null)) }
        probeJob = scope.launch {
            val probe = engine.probe()
            val phase = when (probe.status) {
                FeatureStatus.AVAILABLE -> ModelPhase.READY
                FeatureStatus.DOWNLOADABLE, FeatureStatus.DOWNLOADING -> ModelPhase.DOWNLOADABLE
                else -> if (probe.error != null) ModelPhase.ERROR else ModelPhase.UNAVAILABLE
            }
            mutate {
                it.copy(
                    model = ModelUi(
                        phase = phase,
                        label = probe.label,
                        baseModel = probe.baseModel,
                        status = probe.status,
                        detail = probe.detail ?: probe.error,
                        fallback = probe.fallback,
                        error = if (phase == ModelPhase.ERROR) probe.error else null,
                        options = probe.options,
                        supportsTools = probe.supportsTools,
                    ),
                )
            }
            if (probe.baseModel != null) onProbeResolvedModel()
            if (phase == ModelPhase.READY) warmup()
        }
    }

    /**
     * Quiet background warmup: raises the accelerator so the first user turn
     * avoids the 1.5–3s cold start. Guarded strategies per the warm-up plan:
     * not while generating (real turns own AICore), canceled before send.
     */
    fun warmup() {
        if (!state().warmupEnabled) return
        if (state().model.phase != ModelPhase.READY) return
        if (state().isGenerating) return
        warmupJob?.cancel()
        warmupJob = scope.launch(Dispatchers.IO) {
            engine.warmup()
        }
    }

    fun cancelWarmup() {
        warmupJob?.cancel()
        warmupJob = null
    }

    fun download() {
        if (state().model.phase == ModelPhase.DOWNLOADING) return
        downloadJob?.cancel()
        mutate { it.copy(
            model = it.model.copy(
                phase = ModelPhase.DOWNLOADING,
                downloadedBytes = 0,
                totalBytes = 0,
                error = null,
            ),
        ) }
        downloadJob = scope.launch {
            engine.download().collect { event ->
                when (event) {
                    is DownloadEvent.Started -> mutate { it.copy(
                        model = it.model.copy(
                            phase = ModelPhase.DOWNLOADING,
                            totalBytes = event.totalBytes,
                        ),
                    ) }

                    is DownloadEvent.Progress -> mutate { it.copy(
                        model = it.model.copy(
                            phase = ModelPhase.DOWNLOADING,
                            downloadedBytes = event.downloadedBytes,
                            totalBytes = event.totalBytes,
                        ),
                    ) }

                    is DownloadEvent.Completed -> {
                        mutate { it.copy(
                            model = it.model.copy(
                                phase = ModelPhase.READY,
                                baseModel = event.baseModel ?: it.model.baseModel,
                                error = null,
                            ),
                        ) }
                        postNotice(
                            tr(
                                "${state().model.label} 준비가 끝났어요.",
                                "${state().model.label} is ready.",
                            ),
                        )
                        warmup()
                    }

                    is DownloadEvent.Failed -> mutate { it.copy(
                        model = it.model.copy(
                            phase = ModelPhase.ERROR,
                            error = event.message,
                        ),
                    ) }
                }
            }
        }
    }

    fun select(option: ModelOption) {
        if (state().isGenerating) {
            // Switching the engine closes the active client, which would abort
            // the response that is still streaming. Keep the choice visible now
            // and swap the engine once the current turn ends.
            pendingModel = option
        } else {
            pendingModel = null
            cancelWarmup()
            engine.selectModel(option)
        }
        val phase = when (option.status) {
            FeatureStatus.AVAILABLE -> ModelPhase.READY
            FeatureStatus.DOWNLOADABLE -> ModelPhase.DOWNLOADABLE
            FeatureStatus.DOWNLOADING -> ModelPhase.DOWNLOADING
            else -> if (option.error != null) ModelPhase.ERROR else ModelPhase.UNAVAILABLE
        }
        mutate { it.copy(
            model = it.model.copy(
                phase = phase,
                label = option.label,
                baseModel = option.baseModel,
                status = option.status,
                fallback = option.preference == com.google.mlkit.genai.prompt.ModelPreference.FAST,
                detail = option.error,
                error = if (phase == ModelPhase.ERROR) option.error else null,
                supportsTools = option.supportsTools,
                pendingSelection = state().isGenerating,
            ),
        ) }
    }

    /**
     * Apply a model the user picked while a turn was streaming. The engine
     * switch closes the old client, so it must wait until the turn that owns
     * AICore has really ended.
     */
    fun applyPendingModel() {
        pendingModel?.let {
            pendingModel = null
            engine.selectModel(it)
        }
    }
}
