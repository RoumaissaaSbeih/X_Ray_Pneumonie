const fileInput = document.getElementById("xrayImage");
const imagePreview = document.getElementById("imagePreview");
const uploadEmpty = document.getElementById("uploadEmpty");
const fileStatus = document.getElementById("fileStatus");
const analyzeBtn = document.getElementById("analyzeBtn");
const uploadZone = document.querySelector(".upload-zone");

const validExtensions = ["jpg", "jpeg", "png"];

function updatePreview(file) {
    if (!file) {
        return;
    }

    const extension = file.name.split(".").pop().toLowerCase();
    if (!validExtensions.includes(extension)) {
        fileInput.value = "";
        imagePreview.style.display = "none";
        uploadEmpty.style.display = "grid";
        fileStatus.textContent = "Format invalide. Utilisez JPG, JPEG ou PNG.";
        analyzeBtn.disabled = true;
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        imagePreview.style.display = "block";
        uploadEmpty.style.display = "none";
        fileStatus.textContent = `${file.name} sélectionnée`;
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

if (fileInput) {
    fileInput.addEventListener("change", (event) => {
        updatePreview(event.target.files[0]);
    });
}

if (uploadZone && fileInput) {
    uploadZone.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
        uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (event) => {
        event.preventDefault();
        uploadZone.classList.remove("dragover");
        const file = event.dataTransfer.files[0];
        if (file) {
            fileInput.files = event.dataTransfer.files;
            updatePreview(file);
        }
    });
}

const diagnosisChart = document.getElementById("diagnosisChart");
if (diagnosisChart && window.Chart) {
    const normal = Number(diagnosisChart.dataset.normal || 0);
    const pneumonia = Number(diagnosisChart.dataset.pneumonia || 0);

    new Chart(diagnosisChart, {
        type: "doughnut",
        data: {
            labels: ["NORMAL", "PNEUMONIA"],
            datasets: [{
                data: [normal, pneumonia],
                backgroundColor: ["#16a34a", "#dc2626"],
                borderColor: "#ffffff",
                borderWidth: 4,
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        usePointStyle: true,
                        padding: 18,
                        font: {
                            weight: "700"
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.label}: ${context.raw} analyse(s)`
                    }
                }
            },
            cutout: "68%"
        }
    });
}
