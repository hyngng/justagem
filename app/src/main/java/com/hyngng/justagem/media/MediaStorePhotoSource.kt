package com.hyngng.justagem.media

import android.content.ContentUris
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import com.hyngng.justagem.ui.i18n.tr

/**
 * The MediaStore read side of photo indexing: the device photo count and the
 * ordered photo list with change fingerprints. speak nothing about embeddings;
 * a photo only becomes a [PhotoRecord] once indexed.
 */
class MediaStorePhotoSource(context: Context) {

    private val appContext = context.applicationContext

    data class PhotoItem(
        val id: Long,
        val uri: Uri,
        val displayName: String,
        val dateTaken: Long,
        val width: Int,
        val height: Int,
        val dateModified: Long,
        val size: Long,
    ) {
        val fingerprint: String get() = "$dateModified:$size:$width:$height:$displayName"
    }

    /** Device total, or null when MediaStore is temporarily unreadable. */
    fun photoCount(): Int? {
        val projection = arrayOf(MediaStore.Images.Media._ID)
        return runCatching {
            appContext.contentResolver.query(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                projection,
                null,
                null,
                null,
            )?.use { it.count } ?: error("MediaStore returned no cursor")
        }.getOrNull()
    }

    fun queryPhotos(): List<PhotoItem> {
        val projection = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.DATE_TAKEN,
            MediaStore.Images.Media.WIDTH,
            MediaStore.Images.Media.HEIGHT,
            MediaStore.Images.Media.DATE_MODIFIED,
            MediaStore.Images.Media.SIZE,
        )
        val items = mutableListOf<PhotoItem>()
        val sortOrder = "${MediaStore.Images.Media.DATE_TAKEN} DESC"

        val cursor = appContext.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            projection,
            null,
            null,
            sortOrder,
        ) ?: error("MediaStore returned no cursor")
        cursor.use {
            val idCol = it.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val nameCol = it.getColumnIndex(MediaStore.Images.Media.DISPLAY_NAME)
            val dateCol = it.getColumnIndex(MediaStore.Images.Media.DATE_TAKEN)
            val widthCol = it.getColumnIndex(MediaStore.Images.Media.WIDTH)
            val heightCol = it.getColumnIndex(MediaStore.Images.Media.HEIGHT)
            val modifiedCol = it.getColumnIndex(MediaStore.Images.Media.DATE_MODIFIED)
            val sizeCol = it.getColumnIndex(MediaStore.Images.Media.SIZE)

            while (it.moveToNext()) {
                val id = it.getLong(idCol)
                val name = if (nameCol >= 0) it.getString(nameCol) ?: tr("사진", "Photo") else tr("사진", "Photo")
                val date = if (dateCol >= 0) it.getLong(dateCol) else 0L
                val w = if (widthCol >= 0) it.getInt(widthCol) else 0
                val h = if (heightCol >= 0) it.getInt(heightCol) else 0
                val modified = if (modifiedCol >= 0) it.getLong(modifiedCol) else 0L
                val size = if (sizeCol >= 0) it.getLong(sizeCol) else 0L
                val uri = ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id)
                items.add(PhotoItem(id, uri, name, date, w, h, modified, size))
            }
        }
        return items
    }
}
