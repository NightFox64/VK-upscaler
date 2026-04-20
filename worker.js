self.onerror = (msg, src, line, col, err) => {
  console.error('[worker] uncaught error:', msg, 'at', src, line, col, err);
};

try {
  importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js');
  console.log('[worker] TF.js loaded, version:', tf.version.tfjs);
} catch(e) {
  console.error('[worker] failed to load TF.js:', e);
}

const tasks = new Map();

// Простая CNN для предсказания параметров коррекции
async function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [64, 64, 3], filters: 16, kernelSize: 3, activation: 'relu', padding: 'same' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }));
  model.add(tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  // Выход: [brightness_delta, contrast_factor, saturation_factor]
  model.add(tf.layers.dense({ units: 3, activation: 'tanh' }));
  return model;
}

// Вычисление статистик изображения для эвристики
function computeStats(pixels, width, height) {
  let rSum = 0, gSum = 0, bSum = 0;
  let rSqSum = 0, gSqSum = 0, bSqSum = 0;
  const n = width * height;
  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i] / 255, g = pixels[i + 1] / 255, b = pixels[i + 2] / 255;
    rSum += r; gSum += g; bSum += b;
    rSqSum += r * r; gSqSum += g * g; bSqSum += b * b;
  }
  const rMean = rSum / n, gMean = gSum / n, bMean = bSum / n;
  const rStd = Math.sqrt(rSqSum / n - rMean * rMean);
  const gStd = Math.sqrt(gSqSum / n - gMean * gMean);
  const bStd = Math.sqrt(bSqSum / n - bMean * bMean);
  const brightness = (rMean + gMean + bMean) / 3;
  const contrast = (rStd + gStd + bStd) / 3;
  const saturation = Math.max(rMean, gMean, bMean) - Math.min(rMean, gMean, bMean);
  return { brightness, contrast, saturation };
}

// Ручной bilinear resize RGBA->RGB до 64x64 (без tf.browser, работает в Worker)
function resizeTo64(pixels, srcW, srcH) {
  const OUT = 64;
  const rgb = new Float32Array(OUT * OUT * 3);
  const xRatio = srcW / OUT, yRatio = srcH / OUT;
  for (let y = 0; y < OUT; y++) {
    for (let x = 0; x < OUT; x++) {
      const sx = Math.min(Math.floor(x * xRatio), srcW - 1);
      const sy = Math.min(Math.floor(y * yRatio), srcH - 1);
      const si = (sy * srcW + sx) * 4;
      const di = (y * OUT + x) * 3;
      rgb[di]     = pixels[si]     / 127.5 - 1;
      rgb[di + 1] = pixels[si + 1] / 127.5 - 1;
      rgb[di + 2] = pixels[si + 2] / 127.5 - 1;
    }
  }
  return rgb;
}

// Предсказание параметров через ML + эвристику
async function predictParams(model, imageData, width, height) {
  const rgb64 = resizeTo64(imageData, width, height);
  const tensor = tf.tensor4d(rgb64, [1, 64, 64, 3]);
  const pred = await model.predict(tensor).data();
  tensor.dispose();

  const stats = computeStats(imageData, width, height);

  // Комбинируем ML-предсказание с эвристикой
  const targetBrightness = 0.45;
  const targetContrast = 0.18;
  const targetSaturation = 0.25;

  const brightnessDelta = (targetBrightness - stats.brightness) * 0.6 + pred[0] * 0.05;
  const contrastFactor = 1 + (targetContrast - stats.contrast) * 1.5 + pred[1] * 0.1;
  const saturationFactor = 1 + (targetSaturation - stats.saturation) * 1.2 + pred[2] * 0.1;

  return {
    brightnessDelta: Math.max(-0.4, Math.min(0.4, brightnessDelta)),
    contrastFactor: Math.max(0.5, Math.min(2.5, contrastFactor)),
    saturationFactor: Math.max(0.5, Math.min(2.5, saturationFactor))
  };
}

// Применение параметров к пикселям
function applyCorrection(pixels, params) {
  const { brightnessDelta, contrastFactor, saturationFactor } = params;
  const result = new Uint8ClampedArray(pixels.length);

  for (let i = 0; i < pixels.length; i += 4) {
    let r = pixels[i] / 255;
    let g = pixels[i + 1] / 255;
    let b = pixels[i + 2] / 255;

    // Яркость
    r += brightnessDelta; g += brightnessDelta; b += brightnessDelta;

    // Контрастность
    r = (r - 0.5) * contrastFactor + 0.5;
    g = (g - 0.5) * contrastFactor + 0.5;
    b = (b - 0.5) * contrastFactor + 0.5;

    // Насыщенность через HSL
    const gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    r = gray + (r - gray) * saturationFactor;
    g = gray + (g - gray) * saturationFactor;
    b = gray + (b - gray) * saturationFactor;

    result[i] = Math.round(Math.max(0, Math.min(1, r)) * 255);
    result[i + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
    result[i + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
    result[i + 3] = pixels[i + 3];
  }
  return result;
}

let model = null;

async function processTask(taskId, imageData, width, height) {
  const task = tasks.get(taskId);
  if (!task) return;

  try {
    task.status = 'processing';
    postMessage({ type: 'status', taskId, status: 'processing', progress: 0 });

    if (!model) {
      postMessage({ type: 'status', taskId, status: 'processing', progress: 10 });
      model = await buildModel();
    }

    postMessage({ type: 'status', taskId, status: 'processing', progress: 30 });

    if (task.aborted) { task.status = 'aborted'; return; }

    const params = await predictParams(model, imageData, width, height);

    postMessage({ type: 'status', taskId, status: 'processing', progress: 70 });

    if (task.aborted) { task.status = 'aborted'; return; }

    const resultPixels = applyCorrection(imageData, params);

    postMessage({ type: 'status', taskId, status: 'processing', progress: 95 });

    task.status = 'done';
    task.result = { pixels: resultPixels, width, height, params };

    postMessage({ type: 'status', taskId, status: 'done', progress: 100 });
    postMessage({ type: 'result', taskId, pixels: resultPixels.buffer, width, height, params }, [resultPixels.buffer]);

  } catch (e) {
    console.error('[worker] error in task', taskId, e);
    task.status = 'error';
    task.error = e.message || String(e);
    postMessage({ type: 'status', taskId, status: 'error', progress: 0, error: e.message || String(e) });
  }
}

self.onmessage = function (e) {
  const { type, taskId } = e.data;

  if (type === 'process') {
    const { pixels, width, height } = e.data;
    tasks.set(taskId, { status: 'queued', aborted: false });
    processTask(taskId, new Uint8ClampedArray(pixels), width, height);

  } else if (type === 'abort') {
    const task = tasks.get(taskId);
    if (task && task.status !== 'done') {
      task.aborted = true;
      task.status = 'aborted';
      postMessage({ type: 'abort', taskId, success: true });
    } else {
      postMessage({ type: 'abort', taskId, success: false });
    }

  } else if (type === 'getStatus') {
    const task = tasks.get(taskId);
    postMessage({ type: 'statusResponse', taskId, status: task ? task.status : 'unknown' });
  }
};
