
import streamlit as st
import cv2
import os
import numpy as np
import csv
import pandas as pd
from datetime import datetime


# ============================================================
# Haar Cascade - FIXED
# ============================================================

cascade_filename = "haarcascade_frontalface_default.xml"

# First try OpenCV's built-in haarcascade folder
cascade_path = os.path.join(
    cv2.data.haarcascades,
    cascade_filename
)

# If not found, try same folder as this Python file
if not os.path.exists(cascade_path):
    cascade_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        cascade_filename
    )

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    st.error(
        f"""
        ❌ Haar Cascade could not be loaded.

        Searched at:
        {cascade_path}

        Please make sure this file exists:
        haarcascade_frontalface_default.xml
        """
    )
    st.stop()


# ============================================================
# Paths
# ============================================================

dataset_path = "captured_faces"
model_path = "face_model.yml"
label_map_path = "label_map.npy"

# Create dataset folder automatically
os.makedirs(dataset_path, exist_ok=True)


# ============================================================
# Function: Capture New Student Data
# ============================================================

def capture_student_data(student_name):

    student_name = student_name.strip()

    if not student_name:
        st.warning("⚠️ Please enter student name.")
        return

    output_dir = os.path.join(
        dataset_path,
        student_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error(
            "❌ Camera could not be opened. "
            "Please check camera permission."
        )
        return

    count = 0
    stframe = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            st.error("❌ Could not read camera frame.")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            if count >= 100:
                break

            face_img = gray[
                y:y + h,
                x:x + w
            ]

            if face_img.size == 0:
                continue

            # Resize image before saving
            face_img = cv2.resize(
                face_img,
                (200, 200)
            )

            count += 1

            file_name = os.path.join(
                output_dir,
                f"{student_name}_{count}.jpg"
            )

            cv2.imwrite(
                file_name,
                face_img
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"Captured: {count}/100",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        stframe.image(
            rgb_frame,
            channels="RGB"
        )

        if count >= 100:
            break

    cap.release()
    stframe.empty()

    if count >= 100:

        st.success(
            f"✅ 100 images captured for {student_name}"
        )

    else:

        st.warning(
            f"⚠️ Only {count} images captured."
        )


# ============================================================
# Function: Retrain Model
# ============================================================

def retrain_model():

    faces = []
    labels = []
    label_map = {}

    current_label = 0

    # Make sure dataset exists
    os.makedirs(
        dataset_path,
        exist_ok=True
    )

    student_folders = os.listdir(
        dataset_path
    )

    if len(student_folders) == 0:

        st.warning(
            "⚠️ No student data found. "
            "Please add a student first."
        )

        return

    # --------------------------------------------------------
    # Read every student's folder
    # --------------------------------------------------------

    for user_name in sorted(student_folders):

        user_folder = os.path.join(
            dataset_path,
            user_name
        )

        if not os.path.isdir(user_folder):
            continue

        student_image_count = 0

        for file in os.listdir(user_folder):

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            img_path = os.path.join(
                user_folder,
                file
            )

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            try:

                img = cv2.resize(
                    img,
                    (200, 200)
                )

            except Exception:
                continue

            faces.append(img)

            labels.append(
                current_label
            )

            student_image_count += 1

        # Only create label if images exist
        if student_image_count > 0:

            label_map[
                current_label
            ] = user_name

            current_label += 1

    # --------------------------------------------------------
    # Check training data
    # --------------------------------------------------------

    if len(faces) == 0:

        st.error(
            """
            ❌ No valid face images found.

            Please first:
            1. Add New Student
            2. Capture Data
            3. Then Retrain Model
            """
        )

        return

    if len(labels) == 0:

        st.error(
            "❌ No labels found for training."
        )

        return

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    try:

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.train(
            faces,
            np.array(
                labels,
                dtype=np.int32
            )
        )

        recognizer.save(
            model_path
        )

        np.save(
            label_map_path,
            label_map
        )

        st.success(
            "✅ Model retrained successfully "
            "with new student data!"
        )

        st.info(
            f"👨‍🎓 Students: {len(label_map)} | "
            f"📷 Images: {len(faces)}"
        )

    except AttributeError:

        st.error(
            """
            ❌ cv2.face is not available.

            Install OpenCV Contrib:

            pip uninstall opencv-python -y
            pip install opencv-contrib-python
            """
        )

    except Exception as e:

        st.error(
            f"❌ Model training error:\n\n{e}"
        )


# ============================================================
# Function: Attendance System
# ============================================================

def run_attendance(
    lecture_number,
    subject_name
):

    csv_file = (
        f"attendance_lecture{lecture_number}.csv"
    )

    # --------------------------------------------------------
    # Create CSV if it doesn't exist
    # --------------------------------------------------------

    try:

        with open(
            csv_file,
            "x",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "Name",
                    "Date",
                    "Time",
                    "Lecture",
                    "Subject"
                ]
            )

    except FileExistsError:

        pass

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(model_path):

        st.error(
            """
            ❌ Face model not found.

            Please go to:

            Retrain Model
            ↓
            Retrain Now
            """
        )

        return

    if not os.path.exists(label_map_path):

        st.error(
            """
            ❌ Label map not found.

            Please retrain the model.
            """
        )

        return

    # --------------------------------------------------------
    # Load recognizer
    # --------------------------------------------------------

    try:

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        recognizer.read(
            model_path
        )

        label_map = np.load(
            label_map_path,
            allow_pickle=True
        ).item()

    except Exception as e:

        st.error(
            f"❌ Could not load face model:\n\n{e}"
        )

        return

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        st.error(
            "❌ Camera could not be opened."
        )

        return

    marked_students = set()

    stframe = st.empty()

    st.info(
        "📷 Attendance system started..."
    )

    # --------------------------------------------------------
    # Camera loop
    # --------------------------------------------------------

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:

            st.error(
                "❌ Could not read camera frame."
            )

            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face_roi = gray[
                y:y + h,
                x:x + w
            ]

            if face_roi.size == 0:
                continue

            roi = cv2.resize(
                face_roi,
                (200, 200)
            )

            try:

                label_id, confidence = recognizer.predict(
                    roi
                )

            except Exception:

                continue

            # ------------------------------------------------
            # Recognize face
            # ------------------------------------------------

            if (
                confidence < 50
                and label_id in label_map
            ):

                name = label_map[label_id]

            else:

                name = "Unknown"

            # ------------------------------------------------
            # Mark attendance
            # ------------------------------------------------

            if name != "Unknown":

                if name not in marked_students:

                    now = datetime.now()

                    date_str = now.strftime(
                        "%Y-%m-%d"
                    )

                    time_str = now.strftime(
                        "%H:%M:%S"
                    )

                    with open(
                        csv_file,
                        "a",
                        newline="",
                        encoding="utf-8"
                    ) as f:

                        writer = csv.writer(f)

                        writer.writerow(
                            [
                                name,
                                date_str,
                                time_str,
                                lecture_number,
                                subject_name
                            ]
                        )

                    marked_students.add(
                        name
                    )

                    status = "Marked"

                else:

                    status = "Already Marked"

            else:

                status = "Unknown"

            # ------------------------------------------------
            # Draw rectangle
            # ------------------------------------------------

            if name == "Unknown":

                color = (
                    0,
                    0,
                    255
                )

            else:

                color = (
                    0,
                    255,
                    0
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{name} - {status}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # ----------------------------------------------------
        # Display frame
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        stframe.image(
            rgb_frame,
            channels="RGB"
        )

    cap.release()


# ============================================================
# Streamlit Dashboard
# ============================================================

st.title(
    "📖 Face Recognition Attendance Dashboard"
)

menu = st.sidebar.radio(
    "Select Option",
    [
        "Add New Student",
        "Retrain Model",
        "Attendance System"
    ]
)


# ============================================================
# ADD NEW STUDENT
# ============================================================

if menu == "Add New Student":

    st.header(
        "➕ Add New Student"
    )

    student_name = st.text_input(
        "Enter New Student Name"
    )

    if st.button(
        "Capture Data",
        type="primary"
    ):

        if student_name.strip():

            capture_student_data(
                student_name
            )

        else:

            st.warning(
                "⚠️ Please enter student name."
            )


# ============================================================
# RETRAIN MODEL
# ============================================================

elif menu == "Retrain Model":

    st.header(
        "🔄 Retrain Model"
    )

    if st.button(
        "Retrain Now",
        type="primary"
    ):

        retrain_model()


# ============================================================
# ATTENDANCE SYSTEM
# ============================================================

elif menu == "Attendance System":

    st.header(
        "📝 Attendance System"
    )

    lecture_number = st.text_input(
        "Enter Lecture Number"
    )

    subject_name = st.text_input(
        "Enter Subject Name"
    )

    if st.button(
        "Start Attendance",
        type="primary"
    ):

        if lecture_number and subject_name:

            run_attendance(
                lecture_number,
                subject_name
            )

        else:

            st.warning(
                "⚠️ Please enter Lecture Number "
                "and Subject Name."
            )

    # --------------------------------------------------------
    # Show Attendance Records
    # --------------------------------------------------------

    if st.button(
        "Show Attendance Records"
    ):

        if lecture_number:

            csv_file = (
                f"attendance_lecture{lecture_number}.csv"
            )

            try:

                df = pd.read_csv(
                    csv_file
                )

                st.subheader(
                    "📋 Attendance Records"
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                st.download_button(
                    label="⬇️ Download Attendance CSV",
                    data=df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=csv_file,
                    mime="text/csv"
                )

            except FileNotFoundError:

                st.warning(
                    "⚠️ No attendance file found "
                    "for this lecture."
                )

        else:

            st.warning(
                "⚠️ Please enter Lecture Number."
            )

