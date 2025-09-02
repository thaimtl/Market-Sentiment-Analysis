const App = {
    init() {
        this.setupEventListeners();
        this.setupInputValidation();
        console.info('FinBERT Sentiment Analysis App initialized');
    },

    setupEventListeners() {
        const textAnalyzeBtn = document.getElementById('analyze-text-btn');
        const stockAnalyzeBtn = document.getElementById('analyze-stock-btn');
        
        if (textAnalyzeBtn) {
            textAnalyzeBtn.addEventListener('click', EventHandlers.handleTextAnalysis);
        }
        
        if (stockAnalyzeBtn) {
            stockAnalyzeBtn.addEventListener('click', EventHandlers.handleStockAnalysis);
        }

        document.addEventListener('keypress', EventHandlers.handleKeyboardShortcuts);

        const tickerInput = document.getElementById('ticker-input');
        if (tickerInput) {
            tickerInput.addEventListener('input', EventHandlers.handleTickerInput);
        }
    },

    setupInputValidation() {
        const headlinesInput = document.getElementById('headlines-input');
        const daysInput = document.getElementById('days-input');

        if (headlinesInput) {
            headlinesInput.addEventListener('change', (e) => {
                const value = parseInt(e.target.value);
                if (value < CONFIG.UI.MIN_HEADLINES) e.target.value = CONFIG.UI.MIN_HEADLINES;
                if (value > CONFIG.UI.MAX_HEADLINES) e.target.value = CONFIG.UI.MAX_HEADLINES;
            });
        }

        if (daysInput) {
            daysInput.addEventListener('change', (e) => {
                const value = parseInt(e.target.value);
                if (value < 1) e.target.value = 1;
                if (value > 30) e.target.value = 30;
            });
        }
    }
};

document.addEventListener('DOMContentLoaded', () => {
    App.init();
});