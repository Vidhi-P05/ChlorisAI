// File handling and upload functionality

let selectedFiles = [];

// Initialize drag and drop
document.addEventListener('DOMContentLoaded', function() {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('fileInput');
    
    // Drag and drop handlers
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);
    uploadBox.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
});

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadBox').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadBox').classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadBox').classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFiles(files);
}

function addFiles(files) {
    // Filter image files
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
        alert('Please select image files only (JPG, PNG)');
        return;
    }
    
    // Add to selected files
    selectedFiles = [...selectedFiles, ...imageFiles];
    updateFileList();
    updateUploadButton();
}

function updateFileList() {
    const fileList = document.getElementById('fileList');
    const fileListItems = document.getElementById('fileListItems');
    
    if (selectedFiles.length === 0) {
        fileList.style.display = 'none';
        return;
    }
    
    fileList.style.display = 'block';
    fileListItems.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button onclick="removeFile(${index})" style="background: #e74c3c; color: white; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer;">Remove</button>
        `;
        fileListItems.appendChild(li);
    });
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
    updateUploadButton();
}

function clearFiles() {
    selectedFiles = [];
    document.getElementById('fileInput').value = '';
    updateFileList();
    updateUploadButton();
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('clearBtn').style.display = 'none';
}

function updateUploadButton() {
    const uploadBtn = document.getElementById('uploadBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    uploadBtn.disabled = selectedFiles.length === 0;
    clearBtn.style.display = selectedFiles.length > 0 ? 'inline-block' : 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function uploadImages() {
    if (selectedFiles.length === 0) {
        alert('Please select at least one image');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Create form data
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('images', file);
    });
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        if (response.ok) {
            displayResults(data.results);
        } else {
            alert('Error: ' + (data.error || 'Failed to process images'));
        }
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        alert('Error uploading images: ' + error.message);
    }
}

function displayResults(results) {
    const resultsSection = document.getElementById('resultsSection');
    const resultsContainer = document.getElementById('resultsContainer');
    
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = '';
    
    results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        if (result.error) {
            card.classList.add('error');
            card.innerHTML = `
                <div class="result-header">
                    <span class="result-filename">${result.filename}</span>
                </div>
                <p style="color: #e74c3c;">${result.error}</p>
            `;
        } else {
            const hasPlant = result.has_plant;
            const primary = result.primary_species;
            const alternatives = result.alternatives || [];
            
            let alternativesHtml = '';
            if (alternatives.length > 0) {
                alternativesHtml = `
                    <div class="alternatives">
                        <h4>Alternative Predictions:</h4>
                        ${alternatives.map(alt => `
                            <div class="alternative-item">
                                <div class="alternative-name">
                                    <div class="common-name">${alt.common_name}</div>
                                    <div class="scientific-name">${alt.scientific_name}</div>
                                </div>
                                <div class="alternative-confidence">${(alt.confidence * 100).toFixed(1)}%</div>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            card.innerHTML = `
                <div class="result-header">
                    <span class="result-filename">${result.filename}</span>
                    <span class="plant-detected ${hasPlant ? 'yes' : 'no'}">
                        ${hasPlant ? '✓ Plant Detected' : '? Uncertain'}
                    </span>
                </div>
                <div class="primary-prediction">
                    <h3>Primary Identification</h3>
                    <div class="species-info">
                        <div class="info-item">
                            <div class="info-label">Common Name</div>
                            <div class="info-value">${primary.common_name}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Scientific Name</div>
                            <div class="info-value" style="font-style: italic;">${primary.scientific_name}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">Category</div>
                            <div class="info-value">${primary.category}</div>
                        </div>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-label">
                            <span>Confidence</span>
                            <span>${(primary.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="confidence-fill" style="width: ${primary.confidence * 100}%;">
                            ${(primary.confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
                ${alternativesHtml}
            `;
        }
        
        resultsContainer.appendChild(card);
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

