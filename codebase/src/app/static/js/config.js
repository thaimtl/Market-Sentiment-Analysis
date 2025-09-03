const CONFIG = {
    API: {
        ENDPOINTS: {
            ANALYZE_TEXT: '/analyze_text',
            ANALYZE_TEXTS_BATCH: '/analyze_texts_batch',
            ANALYZE_STOCK: '/analyze_stock',
            PERFORMANCE: '/performance',
            CLEAR_CACHE: '/clear_cache'
        },
        TIMEOUT: 30000
    },
    UI: {
        LOADING_DELAY: 300,
        MAX_HEADLINES: 100,
        MIN_HEADLINES: 5,
        SHOW_PERFORMANCE_METRICS: true
    },
    SENTIMENT: {
        COLORS: {
            positive: 'var(--success)',
            negative: 'var(--danger)',
            neutral: 'var(--neutral)'
        },
        EMOJIS: {
            positive: '🟢',
            negative: '🔴', 
            neutral: '🟡'
        }
    }
};