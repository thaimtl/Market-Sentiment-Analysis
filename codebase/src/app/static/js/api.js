const ApiService = {
    async makeRequest(endpoint, data, method = 'POST') {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), CONFIG.API.TIMEOUT);

        try {
            const response = await fetch(endpoint, {
                method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - please try again');
            }
            
            throw new Error(`Network error: ${error.message}`);
        }
    },

    async analyzeText(text) {
        return this.makeRequest(CONFIG.API.ENDPOINTS.ANALYZE_TEXT, { text });
    },

    async analyzeTextsBatch(texts) {
        return this.makeRequest(CONFIG.API.ENDPOINTS.ANALYZE_TEXTS_BATCH, { texts });
    },

    async analyzeStock(ticker, days, maxHeadlines) {
        return this.makeRequest(CONFIG.API.ENDPOINTS.ANALYZE_STOCK, {
            ticker: ticker.toUpperCase(),
            days: parseInt(days),
            max_headlines: parseInt(maxHeadlines)
        });
    },

    async getPerformanceMetrics() {
        return this.makeRequest(CONFIG.API.ENDPOINTS.PERFORMANCE, {}, 'GET');
    },

    async clearCache() {
        return this.makeRequest(CONFIG.API.ENDPOINTS.CLEAR_CACHE, {});
    }
};