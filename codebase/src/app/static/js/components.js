const UIComponents = {
    createLoadingState: (message) => `<div class="loading">${message}</div>`,

    createError: (message) => `<div class="error">Error: ${message}</div>`,

    createTextResult: (data) => {
        const { sentiment, confidence, probabilities } = data;
        const style = Utils.getSentimentStyle(sentiment);
        
        return `
            <div class="sentiment-card sentiment-${sentiment}">
                <div class="sentiment-header">
                    ${style.emoji} ${sentiment.toUpperCase()} 
                    <span style="font-weight: normal; opacity: 0.9;">
                        (${(confidence * 100).toFixed(1)}% confidence)
                    </span>
                </div>
                <div class="probability-grid">
                    ${Object.entries(probabilities).map(([sent, prob]) => `
                        <div class="probability-item">
                            <div class="probability-value" style="color: ${Utils.getSentimentStyle(sent).color};">
                                ${(prob * 100).toFixed(1)}%
                            </div>
                            <div class="probability-label">${sent}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    },

    createStockResults: (data, requestedCount) => {
        if (data.news_count === 0) {
            return UIComponents.createEmptyState(data.message, requestedCount);
        }

        let html = `
            ${UIComponents.createSummaryGrid(data, requestedCount)}
            ${UIComponents.createPerformanceMetrics(data.performance_metrics)}
            ${UIComponents.createSentimentDistribution(data)}
            ${UIComponents.createNewsList(data, requestedCount)}
        `;
        
        return html;
    },

    createPerformanceMetrics: (performanceData) => {
        if (!CONFIG.UI.SHOW_PERFORMANCE_METRICS || !performanceData) {
            return '';
        }

        const efficiency = performanceData.efficiency || {};
        
        return `
            <div class="card" style="margin-bottom: 1rem; background: #f8fafc;">
                <h4 style="margin-bottom: 0.5rem; color: var(--text-secondary); font-size: 0.9rem;">
                    ⚡ Performance Metrics
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.5rem; font-size: 0.8rem;">
                    <div style="text-align: center;">
                        <div style="font-weight: 600; color: var(--primary);">${efficiency.analysis_time_seconds?.toFixed(2) || 'N/A'}s</div>
                        <div style="color: var(--text-secondary);">Analysis Time</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-weight: 600; color: var(--success);">${efficiency.throughput_headlines_per_second || 'N/A'}/s</div>
                        <div style="color: var(--text-secondary);">Throughput</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-weight: 600; color: var(--warning);">${efficiency.used_batch_processing ? 'Yes' : 'No'}</div>
                        <div style="color: var(--text-secondary);">Batch Processing</div>
                    </div>
                </div>
            </div>
        `;
    },

    createSummaryGrid: (data, requestedCount) => `
        <div class="summary-grid">
            <div class="summary-stat">
                <div class="summary-stat-value" style="color: var(--primary);">${requestedCount}</div>
                <div class="summary-stat-label">Requested</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value" style="color: var(--secondary);">${data.news_count}</div>
                <div class="summary-stat-label">Found</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value" style="color: var(--neutral);">${data.displayed_count}</div>
                <div class="summary-stat-label">Displayed</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value" style="color: ${Utils.getSentimentStyle(data.sentiment_summary.most_common_sentiment).color};">
                    ${data.sentiment_summary.most_common_sentiment.toUpperCase()}
                </div>
                <div class="summary-stat-label">Overall Sentiment</div>
            </div>
        </div>
    `,

    createSentimentDistribution: (data) => `
        <div class="card" style="margin-bottom: 1rem;">
            <h3 style="margin-bottom: 1rem; color: var(--text);">Sentiment Distribution</h3>
            <div class="probability-grid">
                ${Object.entries(data.sentiment_summary.sentiment_distribution).map(([sentiment, stats]) => `
                    <div class="probability-item">
                        <div class="probability-value" style="color: ${Utils.getSentimentStyle(sentiment).color};">
                            ${stats.count}
                        </div>
                        <div class="probability-label">${sentiment} (${stats.percentage}%)</div>
                    </div>
                `).join('')}
            </div>
        </div>
    `,

    createNewsList: (data, requestedCount) => {
        if (data.news_items.length === 0) return '';

        return `
            <div class="card">
                <h3 style="margin-bottom: 1rem; color: var(--text);">
                    Recent Headlines for ${data.ticker}
                    <span style="font-size: 0.8rem; color: var(--text-secondary); font-weight: normal;">
                        (Requested: ${requestedCount}, Found: ${data.news_count}, Displaying: ${data.displayed_count})
                    </span>
                </h3>
                <div class="news-list">
                    ${data.news_items.map(item => UIComponents.createNewsItem(item)).join('')}
                </div>
            </div>
        `;
    },

    createNewsItem: (item) => {
        const style = Utils.getSentimentStyle(item.sentiment);
        const confidence = (item.confidence * 100).toFixed(1);
        
        return `
            <div class="news-item">
                <div class="news-headline">${item.headline}</div>
                <div class="news-meta">
                    <span>${Utils.formatDate(item.date)}</span>
                    <span class="news-sentiment sentiment-${item.sentiment}-badge">
                        ${style.emoji} ${item.sentiment} (${confidence}%)
                    </span>
                </div>
            </div>
        `;
    },

    createEmptyState: (message, requestedCount) => `
        <div class="empty-state">
            <div class="empty-state-icon">📰</div>
            <h3>No Recent News Found</h3>
            <p>${message}</p>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                Requested: ${requestedCount} headlines
            </p>
        </div>
    `
};