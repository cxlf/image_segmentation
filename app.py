import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# load image 
img = cv2.imread("images\hundmitwaffel.png").astype(np.float32)/255.0
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# flatten the image
pixels = img.reshape(-1, 3)

# initialize k
k = int(input("k: "))

# initialize centroids
centroids = pixels[np.random.choice(pixels.shape[0], k, replace=False)]

def k_means(pixels, centroids, k, max_iterations=100):
    start_time = time.perf_counter()
    iteration_count = 0
    for _ in range(max_iterations):
        distances = np.linalg.norm(pixels[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
        iteration_count += 1
        print(iteration_count)
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    end_time = time.perf_counter()
        
    return labels, centroids, iteration_count, end_time - start_time

def analysis(iteration_count, runtime):
    print(f"k: {k}")
    if runtime > 60:
        minutes = int(runtime // 60)
        seconds = runtime % 60
        print(f"iteration count: {iteration_count}\nruntime: {minutes:.0f} min {seconds:.2f} seconds")
    else:
        print(f"iteration count: {iteration_count}\nruntime: {runtime:.2f} seconds")        

labels, final_centroids, iteration_count, runtime = k_means(pixels, centroids, k)
analysis(iteration_count, runtime)

# assign colors to pixels
final_image = final_centroids[labels].reshape(img.shape)

# ensure final_image pixel values are in the correct range (0-255)
final_image = np.clip(final_image, 0, 1) * (255)
final_image = final_image.astype(np.uint8)

# display the final image
plt.imshow(final_image)
plt.show()

#   additional features:

#   //number of iterations
#   //timer
#   elbow method
#   web implementation
#   raise error
#   save images
#   //different colours
