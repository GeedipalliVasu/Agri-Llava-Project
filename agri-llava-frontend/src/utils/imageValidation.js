// Image validation helper functions
export const isGreenishImage = async (file) => {
  return new Promise((resolve) => {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let totalGreen = 0;
      let totalBrightness = 0;
      let sampledCount = 0;
      const totalPixels = canvas.width * canvas.height;

      // Sample pixels for performance. Use a denser sampling than before to improve reliability.
      // Each pixel has 4 values (RGBA). We'll sample every 8 bytes (~every 2 pixels) instead of 16.
      for (let i = 0; i < imageData.length; i += 8) {
        const r = imageData[i];
        const g = imageData[i + 1];
        const b = imageData[i + 2];

        // Calculate brightness
        const brightness = (r + g + b) / 3;
        totalBrightness += brightness;
        sampledCount++;

        // Relaxed green detection to accept a wider range of real leaf photos
        // Accept green when it's slightly higher than red/blue and above a modest threshold
        if ((g > r * 1.05 && g > b * 1.05 && g > 40) || (g > 80 && g >= r && g >= b)) {
          totalGreen++;
        }
      }

      // Calculate average brightness across sampled pixels
      const avgBrightness = sampledCount > 0 ? totalBrightness / sampledCount : 0;

      // Requirements for a valid leaf image (relaxed):
      // 1. At least 8% of sampled pixels should be greenish (lowered from 20%)
      // 2. Average brightness should be between 30 and 230 (wider range)
      // 3. Image should have reasonable dimensions (allow smaller images)
      const isGreenEnough = sampledCount > 0 ? (totalGreen / sampledCount) > 0.08 : false;
      const hasSuitableBrightness = avgBrightness > 30 && avgBrightness < 230;
      const hasReasonableDimensions = img.width >= 100 && img.height >= 100;

      resolve(isGreenEnough && hasSuitableBrightness && hasReasonableDimensions);
    };

    img.onerror = () => resolve(false);
    img.src = URL.createObjectURL(file);
  });
};

export const validateImage = async (file) => {
  // Check file type
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowedTypes.includes(file.type)) {
    throw new Error('Please upload a valid image file (JPEG, PNG, or WebP format)');
  }

  // Check file size (max 5MB)
  const maxSize = 5 * 1024 * 1024; // 5MB
  if (file.size > maxSize) {
    throw new Error('Image size should be less than 5MB');
  }

  // Check if image appears to be a leaf
  const isLeaf = await isGreenishImage(file);
  if (!isLeaf) {
    throw new Error(
      'Please upload a clear image of a leaf or plant. The image should:\n\n' +
      '• Contain primarily green plant material\n' +
      '• Be well-lit (not too dark or bright)\n' +
      '• Have a minimum size of 200x200 pixels\n' +
      '• Show the leaf clearly without too much background'
    );
  }

  return true;
};