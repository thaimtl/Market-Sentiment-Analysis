const EventHandlers = {
    async handleTextAnalysis() {
        const textInput = document.getElementById('text-input');
        const resultsDiv = document.getElementById('text-results');
        const text = textInput.value.trim();

        try {
            UIState.setLoading('textAnalysis', true);
            resultsDiv.innerHTML = UIComponents.createLoadingState('Analyzing sentiment...');

            await new Promise(resolve => setTimeout(resolve, CONFIG.UI.LOADING_DELAY));

            const result = await SentimentAnalyzer.analyzeText(text);
            resultsDiv.innerHTML = UIComponents.createTextResult(result);

        } catch (error) {
            resultsDiv.innerHTML = UIComponents.createError(error.message);
        } finally {
            UIState.setLoading('textAnalysis', false);
        }
    },

    async handleStockAnalysis() {
        const tickerInput = document.getElementById('ticker-input');
        const daysInput = document.getElementById('days-input');
        const headlinesInput = document.getElementById('headlines-input');
        const resultsDiv = document.getElementById('stock-results');

        const ticker = tickerInput.value.trim();
        const days = parseInt(daysInput.value);
        const maxHeadlines = parseInt(headlinesInput.value);

        try {
            UIState.setLoading('stockAnalysis', true);
            resultsDiv.innerHTML = UIComponents.createLoadingState('Fetching and analyzing stock news...');

            await new Promise(resolve => setTimeout(resolve, CONFIG.UI.LOADING_DELAY));

            const result = await SentimentAnalyzer.analyzeStock(ticker, days, maxHeadlines);
            resultsDiv.innerHTML = UIComponents.createStockResults(result, maxHeadlines);

        } catch (error) {
            resultsDiv.innerHTML = UIComponents.createError(error.message);
        } finally {
            UIState.setLoading('stockAnalysis', false);
        }
    },

    handleKeyboardShortcuts(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            
            if (event.target.id === 'text-input') {
                EventHandlers.handleTextAnalysis();
            } else if (['ticker-input', 'days-input', 'headlines-input'].includes(event.target.id)) {
                EventHandlers.handleStockAnalysis();
            }
        }
    },

    handleTickerInput(event) {
        event.target.value = event.target.value.toUpperCase().replace(/[^A-Z]/g, '');
    }
};