import subprocess
import sys
import os

while True:

    os.system("cls")

    print("======================================")
    print("       FACE ATTENDANCE SYSTEM")
    print("======================================")
    print()
    print("1. Register Student")
    print("2. Train Face Model")
    print("3. Start Attendance")
    print("4. View Attendance")
    print("5. Exit")
    print()

    choice = input("Enter your choice: ")

    if choice == "1":

        print("\nOpening Student Registration...")
        subprocess.run([sys.executable, "register_student.py"])

        input("\nPress Enter to continue...")

    elif choice == "2":

        print("\nTraining face recognition model...")
        subprocess.run([sys.executable, "train_model.py"])

        input("\nPress Enter to continue...")

    elif choice == "3":

        print("\nStarting Attendance System...")
        subprocess.run([sys.executable, "attendance_system.py"])

        input("\nPress Enter to continue...")

    elif choice == "4":

        attendance_file = "attendance/attendance.csv"

        if os.path.exists(attendance_file):

            os.system(f'notepad "{attendance_file}"')

        else:

            print("\nNo attendance record found.")

        input("\nPress Enter to continue...")

    elif choice == "5":

        print("\nThank you!")
        break

    else:

        print("\nInvalid choice!")

        input("\nPress Enter to continue...")