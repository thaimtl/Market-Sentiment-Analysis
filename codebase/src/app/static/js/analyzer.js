const SentimentAnalyzer = {
    async analyzeText(text) {
        if (!Utils.validateInput.text(text)) {
            throw new Error('Please enter valid text to analyze');
        }

        const result = await ApiService.analyzeText(text.trim());
        
        if (!result.success) {
            throw new Error(result.error || 'Analysis failed');
        }

        return result;
    },

    async analyzeStock(ticker, days, maxHeadlines) {
        if (!Utils.validateInput.ticker(ticker)) {
            throw new Error('Please enter a valid stock ticker (1-5 letters)');
        }
        
        if (!Utils.validateInput.number(days, 1, 30)) {
            throw new Error('Days must be between 1 and 30');
        }
        
        if (!Utils.validateInput.number(maxHeadlines, CONFIG.UI.MIN_HEADLINES, CONFIG.UI.MAX_HEADLINES)) {
            throw new Error(`Headlines must be between ${CONFIG.UI.MIN_HEADLINES} and ${CONFIG.UI.MAX_HEADLINES}`);
        }

        const result = await ApiService.analyzeStock(ticker, days, maxHeadlines);
        
        if (!result.success) {
            throw new Error(result.error || 'Stock analysis failed');
        }

        return result;
    }
};