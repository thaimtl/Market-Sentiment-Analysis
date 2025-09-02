const Utils = {
    formatDate: (dateString) => {
        if (!dateString) return 'Unknown date';
        
        try {
            const date = new Date(dateString);
            const now = new Date();
            const diffTime = Math.abs(now - date);
            const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
            
            if (diffDays === 1) return 'Yesterday';
            if (diffDays < 7) return `${diffDays} days ago`;
            return date.toLocaleDateString();
        } catch (error) {
            return dateString;
        }
    },

    validateInput: {
        text: (text) => text && text.trim().length > 0,
        ticker: (ticker) => ticker && /^[A-Z]{1,5}$/.test(ticker.trim()),
        number: (num, min, max) => !isNaN(num) && num >= min && num <= max
    },

    getSentimentStyle: (sentiment) => ({
        color: CONFIG.SENTIMENT.COLORS[sentiment] || CONFIG.SENTIMENT.COLORS.neutral,
        emoji: CONFIG.SENTIMENT.EMOJIS[sentiment] || CONFIG.SENTIMENT.EMOJIS.neutral
    })
};