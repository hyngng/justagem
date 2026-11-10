package com.hyngng.justagem.media

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import com.hyngng.justagem.MainActivity
import com.hyngng.justagem.R
import com.hyngng.justagem.ui.i18n.tr

/**
 * Owns the progress notification for a photo indexing pass: channel creation,
 * the dead-run cleanup, the throttled progress updates and the stop action.
 * Reading `IndexingState` is pushes-only (the caller publishes snapshots), so
 * this class has no view of the index itself.
 */
class PhotoIndexNotifier(
    context: Context,
    private val notificationId: Int,
    private val contentRequestCode: Int,
    private val stopRequestCode: Int,
    private val stopAction: String,
) {

    companion object {
        private const val CHANNEL_ID = "photo_indexing"
        private const val UPDATE_INTERVAL_MS = 250L
    }

    private val appContext = context.applicationContext
    private val notifications = appContext.getSystemService(NotificationManager::class.java)

    private var lastNotificationPercent = -1
    private var lastNotificationStatus = ""
    private var lastNotificationAt = 0L

    init {
        notifications.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                tr("사진 색인", "Photo indexing"),
                NotificationManager.IMPORTANCE_LOW,
            )
        )
        // Indexing runs inside this process. If we are being constructed, the
        // process that may have posted this notification is gone, so anything
        // still in the shade belongs to a dead run and can no longer be stopped.
        notifications.cancel(notificationId)
    }

    /**
     * Removes the progress notification once a run reaches a terminal state.
     * A finished or failed run must not leave a stale entry in the shade: the
     * result is already visible in the settings screen and, for a partial run,
     * as a toast.
     */
    fun cancel() {
        lastNotificationAt = 0L
        lastNotificationPercent = -1
        lastNotificationStatus = ""
        notifications.cancel(notificationId)
    }

    fun publish(state: IndexingState, force: Boolean = false) {
        val percent = (state.currentProgress * 100).toInt().coerceIn(0, 100)
        val now = SystemClock.elapsedRealtime()
        if (!force && now - lastNotificationAt < UPDATE_INTERVAL_MS) {
            return
        }
        if (!force && percent == lastNotificationPercent && state.statusMessage == lastNotificationStatus) {
            return
        }
        lastNotificationAt = now
        lastNotificationPercent = percent
        lastNotificationStatus = state.statusMessage

        val total = state.totalDeviceCount
        val countText = if (total > 0) {
            tr(
                "${"%,d".format(state.indexedCount)}/${"%,d".format(total)}장 ($percent%)",
                "${"%,d".format(state.indexedCount)}/${"%,d".format(total)} photos ($percent%)",
            )
        } else {
            tr("${"%,d".format(state.indexedCount)}장", "${"%,d".format(state.indexedCount)} photos")
        }
        // The system controls its own line spacing, so the only way to give the
        // rows room is to keep the collapsed view to one short line and put the
        // detail on separate lines that BigTextStyle shows when expanded. Only
        // in-progress runs reach this point; terminal states cancel the entry.
        val title = if (state.isStopping) {
            tr("사진 색인 중지 중", "Stopping photo indexing")
        } else {
            tr("사진 색인", "Photo indexing")
        }
        val contentText = when {
            state.isStopping -> tr("잠시만 기다려 주세요…", "Just a moment…")
            state.isIndexing && total > 0 -> tr("사진 분석 중 ($percent%)", "Analyzing photos ($percent%)")
            state.isIndexing -> tr("사진 분석 중", "Analyzing photos")
            else -> state.statusMessage
        }
        val bigText = if (state.isIndexing) {
            buildString {
                append(tr("사진 분석 중", "Analyzing photos"))
                append("\n")
                append(countText)
            }
        } else {
            state.statusMessage
        }
        // FLAG_ACTIVITY_SINGLE_TOP with CLEAR_TOP brings the existing MainActivity instance
        // to the foreground rather than destroying and recreating it, ensuring active
        // viewModelScope indexing passes continue running uninterrupted.
        val openIntent = Intent(appContext, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or
                Intent.FLAG_ACTIVITY_CLEAR_TOP or
                Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val notification = NotificationCompat.Builder(appContext, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle(title)
            .setContentText(contentText)
            .setStyle(NotificationCompat.BigTextStyle().bigText(bigText))
            .setContentIntent(
                PendingIntent.getActivity(
                    appContext,
                    contentRequestCode,
                    openIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
                ),
            )
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setOnlyAlertOnce(true)
            .setOngoing(state.isIndexing)
            .setAutoCancel(!state.isIndexing)
            .setProgress(100, percent, state.isIndexing && total == 0)
        if (state.isIndexing) {
            val stopIntent = Intent(appContext, PhotoIndexingNotificationReceiver::class.java)
                .setAction(stopAction)
            notification.addAction(
                android.R.drawable.ic_menu_close_clear_cancel,
                tr("중지", "Stop"),
                PendingIntent.getBroadcast(
                    appContext,
                    stopRequestCode,
                    stopIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
                ),
            )
        }
        val built = notification.build()
        try {
            notifications.notify(notificationId, built)
        } catch (_: SecurityException) { }
    }
}
