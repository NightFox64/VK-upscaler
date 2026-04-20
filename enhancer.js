/**
 * ImageEnhancer — API модуля улучшения изображений
 * События: 'taskStatusChanged'
 */
class ImageEnhancer extends EventTarget {
  constructor() {
    super();
    this._worker = new Worker('worker.js');
    this._tasks = new Map();
    this._pendingCallbacks = new Map();
    this._worker.onmessage = (e) => this._handleMessage(e.data);
  }

  _handleMessage(data) {
    const { type, taskId } = data;

    if (type === 'status') {
      const task = this._tasks.get(taskId);
      if (task) {
        task.status = data.status;
        task.progress = data.progress;
      }
      this.dispatchEvent(new CustomEvent('taskStatusChanged', {
        detail: { taskId, status: data.status, progress: data.progress }
      }));
    }

    if (type === 'result') {
      const task = this._tasks.get(taskId);
      if (task) {
        task.result = { pixels: new Uint8ClampedArray(data.pixels), width: data.width, height: data.height, params: data.params };
        // Диспатчим отдельное событие когда result точно записан
        this.dispatchEvent(new CustomEvent('taskResultReady', { detail: { taskId } }));
      }
    }

    if (type === 'abort') {
      const cb = this._pendingCallbacks.get(`abort_${taskId}`);
      if (cb) { cb(data.success); this._pendingCallbacks.delete(`abort_${taskId}`); }
    }

    if (type === 'statusResponse') {
      const cb = this._pendingCallbacks.get(`status_${taskId}`);
      if (cb) { cb(data.status); this._pendingCallbacks.delete(`status_${taskId}`); }
    }
  }

  /**
   * Поставить задачу на обработку
   * @param {File|Blob|HTMLImageElement|ImageData} source
   * @returns {Promise<string>} taskId
   */
  async enqueue(source) {
    const taskId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
    const { pixels, width, height } = await this._toPixels(source);
    this._tasks.set(taskId, { status: 'queued', progress: 0, result: null });
    this._worker.postMessage({ type: 'process', taskId, pixels: pixels.buffer, width, height }, [pixels.buffer]);
    return taskId;
  }

  /**
   * Получить статус задачи
   * @param {string} taskId
   * @returns {{ status: string, progress: number }}
   */
  getStatus(taskId) {
    const task = this._tasks.get(taskId);
    if (!task) return { status: 'unknown', progress: 0 };
    return { status: task.status, progress: task.progress };
  }

  /**
   * Прервать задачу
   * @param {string} taskId
   * @returns {Promise<boolean>}
   */
  abort(taskId) {
    return new Promise((resolve) => {
      this._pendingCallbacks.set(`abort_${taskId}`, resolve);
      this._worker.postMessage({ type: 'abort', taskId });
    });
  }

  /**
   * Получить готовое изображение в виде Blob
   * @param {string} taskId
   * @param {'image/jpeg'|'image/png'} [format='image/jpeg']
   * @returns {Promise<Blob>}
   */
  async getResult(taskId, format = 'image/jpeg') {
    const task = this._tasks.get(taskId);
    if (!task) throw new Error('Task not found');
    if (!task.result) throw new Error('Result not ready yet');

    const { pixels, width, height } = task.result;
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.putImageData(new ImageData(pixels, width, height), 0, 0);
    return canvas.convertToBlob({ type: format, quality: 0.92 });
  }

  // --- Вспомогательные методы ---

  async _toPixels(source) {
    if (source instanceof ImageData) {
      return { pixels: new Uint8ClampedArray(source.data), width: source.width, height: source.height };
    }
    if (source instanceof File || source instanceof Blob) {
      source = await this._blobToImageBitmap(source);
    }
    if (source instanceof ImageBitmap || source instanceof HTMLImageElement || source instanceof HTMLCanvasElement) {
      return this._bitmapToPixels(source);
    }
    throw new Error('Unsupported source type');
  }

  async _blobToImageBitmap(blob) {
    // HEIC конвертация
    if (blob.type === 'image/heic' || blob.name?.toLowerCase().endsWith('.heic')) {
      if (typeof heic2any === 'undefined') throw new Error('heic2any not loaded');
      blob = await heic2any({ blob, toType: 'image/jpeg' });
    }
    return createImageBitmap(blob);
  }

  _bitmapToPixels(bitmap) {
    const MAX_PX = 15_000_000;
    let { width, height } = bitmap;
    if (width * height > MAX_PX) {
      const scale = Math.sqrt(MAX_PX / (width * height));
      width = Math.floor(width * scale);
      height = Math.floor(height * scale);
    }
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    return { pixels: new Uint8ClampedArray(imageData.data), width, height };
  }
}

window.ImageEnhancer = ImageEnhancer;
