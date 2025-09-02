const CONFIG = {
    API: {
        ENDPOINTS: {
            ANALYZE_TEXT: '/analyze_text',
            ANALYZE_STOCK: '/analyze_stock'
        },
        TIMEOUT: 30000
    },
    UI: {
        LOADING_DELAY: 300,
        MAX_HEADLINES: 100,
        MIN_HEADLINES: 5
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