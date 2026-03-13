// This file contains JavaScript functions for handling admin-related actions, such as form submissions and dynamic content updates.

document.addEventListener("DOMContentLoaded", function() {
    const employeeForm = document.getElementById("employeeForm");
    const employeeTable = document.getElementById("employeeTable");

    if (employeeForm) {
        employeeForm.addEventListener("submit", function(event) {
            event.preventDefault();
            const formData = new FormData(employeeForm);
            const actionUrl = employeeForm.dataset.action; // URL for add/edit action

            fetch(actionUrl, {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === "success") {
                    alert(data.message);
                    window.location.reload(); // Reload to see updated employee list
                } else {
                    alert(data.message);
                }
            })
            .catch(error => {
                console.error("Error:", error);
                alert("An error occurred while processing your request.");
            });
        });
    }

    if (employeeTable) {
        employeeTable.addEventListener("click", function(event) {
            if (event.target.classList.contains("edit-button")) {
                const employeeId = event.target.dataset.id;
                fetch(`/admin/employees/${employeeId}`)
                .then(response => response.json())
                .then(data => {
                    if (data) {
                        populateForm(data);
                    }
                })
                .catch(error => console.error("Error:", error));
            }
        });
    }

    function populateForm(data) {
        document.getElementById("username").value = data.username;
        document.getElementById("employeeId").value = data.employee_id;
        document.getElementById("mobile").value = data.mobile;
        document.getElementById("authorization").value = data.authorization;
        document.getElementById("role").value = data.role;
        document.getElementById("employeeForm").dataset.action = `/admin/employees/edit/${data.id}`;
    }
});