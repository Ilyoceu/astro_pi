import cv2
import math
from picamzero import Camera
import os

home_dir = os.environ['HOME']
cam = Camera()

def calculate_features(image_cv, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoint, descriptor = orb.detectAndCompute(image_cv, None)
    return keypoint, descriptor

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key = lambda x: x.distance)
    return matches

def find_matching_coordinates(keypoints_1, keypoints_2, matches, image_shape, radius_ratio=0.25):
    h, w = image_shape
    cx, cy = w / 2, h / 2
    max_radius = min(w, h) * radius_ratio

    coordinates_1 = []
    coordinates_2 = []

    for match in matches:
        (x1, y1) = keypoints_1[match.queryIdx].pt
        (x2, y2) = keypoints_2[match.trainIdx].pt

        # keep only features near image center because of camera tilt etc.
        if math.hypot(x1 - cx, y1 - cy) < max_radius:
            coordinates_1.append((x1, y1))
            coordinates_2.append((x2, y2))

    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2, keep_ratio=0.5):
    distances = []
    merged_coordinates = zip(coordinates_1, coordinates_2)
    for (x1, y1), (x2, y2) in merged_coordinates:
        distance = math.hypot(x1 - x2, y1 - y2)
        distances.append(distance)
    if not distances:
        return 0  # avoid division by zero
    distances.sort()
    cutoff = int(len(distances) * (1 - keep_ratio))
    filtered = distances[cutoff:]
    if not filtered:
        return 0  # avoid division by zero
    return sum(filtered) / len(filtered)


def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

# edit variable for amount of pictures min. 3 (capture_sequence logic) max. 42 (project requirement)
# lower pictures are better for accuracy
total_pictures = 42
# time difference in seconds 2 seconds is best
# large time differences causes dissapearing features, small time differences has problems with ORB features
time_difference = 2.0

cam.capture_sequence(f"{home_dir}/sequence.jpg", num_images=total_pictures, interval=time_difference)
images = [f"{home_dir}/sequence-{i:02d}.jpg" for i in range(1, total_pictures + 1)]
images_cv = [cv2.imread(image, 0) for image in images] # Create Opencv image objects

keypoints = []
descriptors = []
for image in images_cv:
    keypoint, descriptor = calculate_features(image, 1000) # Get keypoints and descriptors
    keypoints.append(keypoint)
    descriptors.append(descriptor)

coordinates = []
for i in range(total_pictures - 1):
    matches = calculate_matches(descriptors[i], descriptors[i+1]) # Match descriptors
    coordinates_1, coordinates_2 =  find_matching_coordinates(keypoints[i],keypoints[i+1],matches,images_cv[i].shape)
    coordinates.append((coordinates_1, coordinates_2)) # store the pair

pair_speeds = []
for i in range(total_pictures - 1): 
    coordinates_1, coordinates_2 = coordinates[i]
    feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    speed = calculate_speed_in_kmps(feature_distance, 12648, time_difference)
    pair_speeds.append(speed)

pair_speeds.sort()
speed = pair_speeds[len(pair_speeds) // 2] # Get median speed for best accuracy

# format speed to 4 decimals
speed_formatted = "{:.4f}".format(speed)

# Create a string to write to the file
output_string = speed_formatted

# Write to the file
file_path = "result.txt" 
with open(file_path, 'w') as file:
    file.write(output_string)

result = speed - 7.66
percentage_error = (abs(result) / 7.66) * 100
print("percentage error:", percentage_error)
