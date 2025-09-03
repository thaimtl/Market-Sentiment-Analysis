import re
from typing import List

class BaseValidator:
    """Base validator class"""
    
    def __init__(self, data):
        self.data = data or {}
        self.errors = []
        self._validate()
    
    def _validate(self):
        """Override in subclasses"""
        pass
    
    def is_valid(self):
        """Check if validation passed"""
        return len(self.errors) == 0
    
    def get_error(self):
        """Get first validation error"""
        return self.errors[0] if self.errors else None
    
    def get_errors(self):
        """Get all validation errors"""
        return self.errors

class TextAnalysisValidator(BaseValidator):
    """Validator for text analysis requests"""
    
    def _validate(self):
        self.text = self.data.get('text', '').strip()
        
        if not self.text:
            self.errors.append('No text provided')
        elif len(self.text) > 10000:  # Reasonable limit
            self.errors.append('Text too long (max 10,000 characters)')

class BatchAnalysisValidator(BaseValidator):
    """Validator for batch text analysis requests"""
    
    def _validate(self):
        self.texts = self.data.get('texts', [])
        
        if not isinstance(self.texts, list):
            self.errors.append('Texts must be provided as a list')
            return
        
        if not self.texts:
            self.errors.append('No texts provided')
            return
        
        if len(self.texts) > 100:  # Reasonable batch limit
            self.errors.append('Too many texts (max 100 per batch)')
            return
        
        # Validate each text
        valid_texts = []
        for i, text in enumerate(self.texts):
            if not isinstance(text, str):
                self.errors.append(f'Text at index {i} must be a string')
                continue
            
            text = text.strip()
            if len(text) > 10000:
                self.errors.append(f'Text at index {i} too long (max 10,000 characters)')
                continue
            
            valid_texts.append(text)
        
        self.texts = valid_texts

class StockAnalysisValidator(BaseValidator):
    """Validator for stock analysis requests"""
    
    def _validate(self):
        # Validate ticker
        self.ticker = self.data.get('ticker', '').strip().upper()
        if not self.ticker:
            self.errors.append('No ticker provided')
        elif not re.match(r'^[A-Z]{1,5}$', self.ticker):
            self.errors.append('Invalid ticker format (1-5 letters only)')
        
        # Validate days
        try:
            self.days = int(self.data.get('days', 7))
            if self.days < 1 or self.days > 30:
                self.errors.append('Days must be between 1 and 30')
        except (ValueError, TypeError):
            self.errors.append('Invalid days value')
        
        # Validate max_headlines
        try:
            self.max_headlines = int(self.data.get('max_headlines', 20))
            if self.max_headlines < 5 or self.max_headlines > 100:
                self.errors.append('Max headlines must be between 5 and 100')
        except (ValueError, TypeError):
            self.errors.append('Invalid max headlines value')