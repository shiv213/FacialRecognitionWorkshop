import face_recognition
import cv2 as cv
import numpy as np
import glob
import os

# all photos are in a folder called faces, where file names are names of people in image
faces_encodings = []
faces_names = []  # file names (names of people)

cur_dir = os.getcwd()

path = os.path.join(cur_dir, 'faces/')

list_of_files = [f for f in glob.glob(path + '*.jpg')]  # files are of the same type

number_files = len(list_of_files)

names = list_of_files.copy()

for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])
    # Create array of known names
    names[i] = names[i].replace(cur_dir, "")
    names[i] = names[i].replace("\\faces\\", "")
    names[i] = names[i].replace(".jpg", "")
    names[i] = names[i].replace("_", " ")
    faces_names.append(names[i])

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

video_capture = cv.VideoCapture(0)

while True:
    # grab a single frame of video
    ret, frame = video_capture.read()

    # resize frame to 1/4 of the original size for faster processing
    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # convert image from OpenCV's BGR to RGB for the face_recognition library
    rgb_small_frame = small_frame[:, :, ::-1]

    # only process every other frame of the video to save time
    if process_this_frame:
        # find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # see if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # scale face locations back up since the frame was scaled down to 1/4 it's size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # Draw a rectangle around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Input text label with a name below the face
        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv.destroyAllWindows()
