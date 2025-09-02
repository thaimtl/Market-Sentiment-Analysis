const UIState = {
    isLoading: {
        textAnalysis: false,
        stockAnalysis: false
    },

    setLoading: (type, loading) => {
        UIState.isLoading[type] = loading;
        UIState.updateButtonState(type, loading);
    },

    updateButtonState: (type, loading) => {
        const buttonId = type === 'textAnalysis' ? 'analyze-text-btn' : 'analyze-stock-btn';
        const button = document.getElementById(buttonId);
        
        if (!button) return;

        if (loading) {
            button.disabled = true;
            button.classList.add('btn-loading');
            button.innerHTML = '<span>Analyzing...</span>';
        } else {
            button.disabled = false;
            button.classList.remove('btn-loading');
            button.innerHTML = UIState.getOriginalButtonContent(type);
        }
    },

    getOriginalButtonContent: (type) => {
        const iconSvg = type === 'textAnalysis' 
            ? '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>'
            : '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>';
        
        const buttonText = type === 'textAnalysis' ? 'Analyze Sentiment' : 'Analyze Stock Sentiment';
        
        return `
            <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                ${iconSvg}
            </svg>
            ${buttonText}
        `;
    }
};