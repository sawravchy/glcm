import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops

import cv2


PATCH_SIZE = 80

image = cv2.imread('road.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

road_locations = [(400, 200), (400, 300), (400, 400), (400, 500)]
road_patches = []
for loc in road_locations:
    road_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

sky_locations = [(100, 300), (100, 400), (100, 500), (100, 600)]
sky_patches = []
for loc in sky_locations:
    sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

xs = []
ys = []
for patch in (road_patches + sky_patches):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
          vmin=0, vmax=255)
for (y, x) in road_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(road_patches)], ys[:len(road_patches)], 'go',
        label='Road')
ax.plot(xs[len(road_patches):], ys[len(road_patches):], 'bo',
        label='Sky')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

for i, patch in enumerate(road_patches):
    ax = fig.add_subplot(3, len(road_patches), len(road_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Road %d' % (i + 1))

for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
              vmin=0, vmax=255)
    ax.set_xlabel('Sky %d' % (i + 1))


fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()
