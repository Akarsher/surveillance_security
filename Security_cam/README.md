# Security Camera System

## Overview
The Security Camera System is a FastAPI application designed for monitoring and managing security cameras. It includes features for facial recognition, employee management, and event logging.

## Project Structure
```
Security_cam
├── src
│   ├── app.py                  # Main entry point of the FastAPI application
│   ├── db.py                   # Database setup and session management
│   ├── cloudflare_tunnel.py    # Manages Cloudflare tunnel for remote access
│   └── models
│       └── employee.py         # Defines the Employee model and related methods
├── templates
│   ├── admin_login.html        # Admin login page
│   ├── admin_dashboard.html     # Admin dashboard for managing employees and logs
│   ├── admin_employees.html     # Displays list of employees with management options
│   ├── admin_employee_form.html  # Form for adding/editing employee details
│   ├── admin_logs.html          # Displays logs of security events
│   ├── stream.html              # Structure for viewing camera stream
│   └── employee_profile.html     # Allows employees to view/edit their profiles
├── static
│   ├── css
│   │   └── styles.css          # CSS styles for the application
│   └── js
│       └── admin.js            # JavaScript functions for admin actions
├── known_faces
│   ├── authorized               # Images of authorized personnel
│   └── restricted               # Images of restricted personnel
├── snapshots                    # Directory for storing snapshot images
├── logs                         # Directory for log files
├── requirements.txt             # Python dependencies for the project
└── README.md                    # Documentation for the project
```

## Features
- **Employee Management**: Admin can add, edit, and delete employee records, including username, employee ID, password, mobile number, authorization, role, and photo.
- **Facial Recognition**: The system uses facial recognition to identify authorized and restricted personnel.
- **Event Logging**: Logs security events and actions taken by the admin.
- **Remote Access**: Utilizes Cloudflare tunnel for secure remote access to the application.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd Security_cam
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/app.py
   ```

4. Access the application at `http://localhost:8000` or use the Cloudflare tunnel for remote access.

## Usage
- Navigate to the admin login page to manage employees and view logs.
- Employees can view and edit their profiles, including changing passwords.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.