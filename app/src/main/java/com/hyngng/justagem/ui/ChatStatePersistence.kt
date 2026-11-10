package com.hyngng.justagem.ui

import android.app.Application
import com.hyngng.justagem.data.SessionStore
import com.hyngng.justagem.ui.i18n.AppLanguage
import com.hyngng.justagem.ui.i18n.tr
import com.hyngng.justagem.ui.i18n.updateProcessLanguage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Owns the SessionStore round trip: the debounced writer job, the load/restore
 * gate, and the final synchronous save in onCleared. It reads `ChatUiState`
 * only through [state] and writes through [mutate], so the ViewModel stays the
 * single owner of state.
 */
class ChatStatePersistence(
    application: Application,
    private val scope: CoroutineScope,
    private val state: () -> ChatUiState,
    private val mutate: ((ChatUiState) -> ChatUiState) -> Unit,
    /** Posts a user-facing notice (runs on the main thread). */
    private val postNotice: (String) -> Unit,
    /**
     * Runs after a restore has been applied and the process language mirrored,
     * e.g. canceling a warmup the stored preference says should be off.
     */
    private val postRestore: () -> Unit,
) {

    private val store = SessionStore(application)

    private val persistLock = Any()
    private var persistFailureNoticeShown = false
    private var pendingPersist: SessionStore.PersistedState? = null
    private var persistJob: Job? = null
    private var persistedStateLoaded = false

    /**
     * The field list of a disk snapshot lives here so regular saves and the
     * onCleared fallback can never drift apart. lastBaseModel stays
     * caller-controlled: the fallback runs when nothing was ever loaded, so it
     * must not record a model that was never resolved.
     */
    private fun snapshot(lastBaseModel: String?) =
        SessionStore.PersistedState(
            sessions = state().sessions,
            activeId = state().activeId,
            hapticsEnabled = state().hapticsEnabled,
            soundsEnabled = state().soundsEnabled,
            warmupEnabled = state().warmupEnabled,
            languageTag = state().languageTag,
            prePrompt = state().prePrompt,
            thinkingEnabled = state().thinkingEnabled,
            temperature = state().temperature,
            topK = state().topK,
            maxOutputTokens = state().maxOutputTokens,
            lastBaseModel = lastBaseModel,
        )

    /**
     * Loads the store off the main thread and applies it once. Stored values
     * lose to values the user changed in this session before the load landed;
     * sessions/messages only appear when nothing was created yet.
     */
    fun restore() {
        scope.launch(Dispatchers.IO) {
            val persisted = store.load()
            val savedSessions = persisted.sessions.filter { it.messages.isNotEmpty() }
            withContext(Dispatchers.Main.immediate) {
                if (!persistedStateLoaded) {
                    val defaults = ChatUiState()
                    val current = state()
                    mutate { ui ->
                        ui.copy(
                            sessions = if (ui.sessions.isEmpty()) savedSessions else ui.sessions,
                            activeId = ui.activeId,
                            messages = ui.messages,
                            hapticsEnabled = if (ui.hapticsEnabled == defaults.hapticsEnabled) persisted.hapticsEnabled else ui.hapticsEnabled,
                            soundsEnabled = if (ui.soundsEnabled == defaults.soundsEnabled) persisted.soundsEnabled else ui.soundsEnabled,
                            warmupEnabled = if (ui.warmupEnabled == defaults.warmupEnabled) persisted.warmupEnabled else ui.warmupEnabled,
                            languageTag = if (ui.languageTag == defaults.languageTag) persisted.languageTag else ui.languageTag,
                            prePrompt = if (ui.prePrompt == defaults.prePrompt) persisted.prePrompt else ui.prePrompt,
                            thinkingEnabled = if (ui.thinkingEnabled == defaults.thinkingEnabled) persisted.thinkingEnabled else ui.thinkingEnabled,
                            temperature = if (ui.temperature == defaults.temperature) persisted.temperature else ui.temperature,
                            topK = if (ui.topK == defaults.topK) persisted.topK else ui.topK,
                            maxOutputTokens = if (ui.maxOutputTokens == defaults.maxOutputTokens) persisted.maxOutputTokens else ui.maxOutputTokens,
                            model = if (ui.model.baseModel == null && !persisted.lastBaseModel.isNullOrBlank()) {
                                ui.model.copy(baseModel = persisted.lastBaseModel)
                            } else {
                                ui.model
                            },
                        )
                    }
                    persistedStateLoaded = true
                    // Mirror the restored language for non-composable notices.
                    updateProcessLanguage(AppLanguage.resolve(state().languageTag))
                    postRestore()
                }
            }
        }
    }

    /** Queues a state snapshot; a single writer drains the queue on IO. */
    fun scheduleSave() {
        if (!persistedStateLoaded) return
        val queued = snapshot(lastBaseModel = state().model.baseModel)
        synchronized(persistLock) {
            pendingPersist = queued
            if (persistJob != null) return
            persistJob = scope.launch(Dispatchers.IO) {
                while (true) {
                    val next = synchronized(persistLock) {
                        val value = pendingPersist
                        pendingPersist = null
                        if (value == null) {
                            persistJob = null
                        }
                        value
                    } ?: return@launch
                    val saved = store.save(next)
                    val notifyFailure = synchronized(persistLock) {
                        when {
                            saved -> {
                                persistFailureNoticeShown = false
                                false
                            }
                            else -> {
                                val first = !persistFailureNoticeShown
                                persistFailureNoticeShown = true
                                first
                            }
                        }
                    }
                    if (notifyFailure) {
                        withContext(Dispatchers.Main.immediate) {
                            postNotice(tr("대화를 저장하지 못했어요. 저장 공간을 확인해 주세요.", "Could not save the conversation. Check available storage."))
                        }
                    }
                }
            }
        }
    }

    /**
     * The synchronous last save before the object dies: anything queued but
     * not yet written wins; otherwise the current state is saved with no
     * baseModel recorded (it may have never been resolved).
     */
    fun saveNow() {
        val finalSnapshot = synchronized(persistLock) {
            pendingPersist ?: snapshot(lastBaseModel = null)
        }
        store.save(finalSnapshot)
    }

    fun close() {
        // SessionStore holds no resources; nothing to release.
    }
}
