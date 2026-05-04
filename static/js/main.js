document.addEventListener('DOMContentLoaded', () => {
    const symptomsInput = document.getElementById('symptoms-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingState = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const suggestionChips = document.querySelectorAll('.suggestion-chip');

    // Handle suggestion chip clicks
    suggestionChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const currentVal = symptomsInput.value.trim();
            const textToAdd = chip.textContent;
            
            if (currentVal) {
                if (!currentVal.includes(textToAdd)) {
                    symptomsInput.value = `${currentVal}, ${textToAdd}`;
                }
            } else {
                symptomsInput.value = textToAdd;
            }
            symptomsInput.focus();
        });
    });

    // Handle Analyze button action
    analyzeBtn.addEventListener('click', () => {
        const symptoms = symptomsInput.value.trim();
        
        if (!symptoms) {
            alert('Please enter your symptoms into the text area first.');
            return;
        }

        fetchRecommendation(symptoms);
    });

    // Make API Call
    async function fetchRecommendation(symptoms) {
        // Toggle view states
        loadingState.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        resultContainer.innerHTML = '';
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symptoms })
            });

            const data = await response.json();
            
            // Artificial delay to mimic deep NLP processing and show animations
            setTimeout(() => {
                loadingState.classList.add('hidden');
                analyzeBtn.disabled = false;
                renderResult(data);
            }, 1000);

        } catch (error) {
            console.error('Error details:', error);
            loadingState.classList.add('hidden');
            analyzeBtn.disabled = false;
            
            resultContainer.innerHTML = `
                <div class="error-card">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <h3>Connection Error</h3>
                    <p>Failed to communicate safely with the prediction core.</p>
                </div>
            `;
            resultContainer.classList.remove('hidden');
        }
    }

    // Process and display server output
    function renderResult(data) {
        let html = '';

        if (data.found === true) {
            const med = data.data;
            html = `
                <div class="result-card">
                    <div class="result-header">
                        <div>
                            <span style="color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Recommended Medicine</span>
                            <div class="medicine-name">${med.medicine}</div>
                        </div>
                        <div class="match-score">
                            <i class="fa-solid fa-check"></i> ${med.score}% Match
                        </div>
                    </div>
                    <div class="result-body">
                        <div class="info-group">
                            <h3><i class="fa-solid fa-circle-info"></i> Description</h3>
                            <p>${med.description}</p>
                        </div>
                        <div class="info-group">
                            <h3><i class="fa-solid fa-shield-halved"></i> Precautions</h3>
                            <p>${med.precautions}</p>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Null state / Not found handling execution
            html = `
                <div class="error-card">
                    <i class="fa-solid fa-circle-xmark"></i>
                    <h3>No Reliable Match Found</h3>
                    <p>${data.message || data.error}</p>
                </div>
            `;
        }

        resultContainer.innerHTML = html;
        resultContainer.classList.remove('hidden');
        
        // Scroll smoothly to output
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
