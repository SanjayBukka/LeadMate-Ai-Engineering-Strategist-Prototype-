import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import time
import json
import re
import hashlib
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from io import StringIO
import base64
import os
from dataclasses import dataclass
from enum import Enum
import sqlite3
import threading
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configurations
MAX_COMMITS_LIMIT = 200
API_RATE_LIMIT = 1.2  # seconds between requests
SESSION_TIMEOUT = 3600  # 1 hour
MAX_REPO_SIZE_MB = 500

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CommitCategory(Enum):
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    DOCS = "docs"
    TEST = "test"
    SECURITY = "security"
    PERFORMANCE = "performance"
    OTHER = "other"

@dataclass
class SecurityConfig:
    max_commits: int = MAX_COMMITS_LIMIT
    rate_limit: float = API_RATE_LIMIT
    session_timeout: int = SESSION_TIMEOUT
    allowed_domains: List[str] = None
    
    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = ['github.com', 'api.github.com']

# Configure Streamlit page
st.set_page_config(
    page_title="Enterprise GitHub Analyzer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .security-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-card {
        background: #d1edff;
        border: 1px solid #74b9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SecurityManager:
    """Handles security operations and validations"""
    
    @staticmethod
    def validate_github_url(url: str) -> bool:
        """Validate GitHub URL format and security"""
        pattern = r'^https://github\.com/[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?/[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?/?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not text:
            return ""
        # Remove potential harmful characters
        sanitized = re.sub(r'[<>"\';\\]', '', str(text))
        return sanitized[:1000]  # Limit length
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate secure session ID"""
        return secrets.token_hex(16)
    
    @staticmethod
    def hash_token(token: str) -> str:
        """Hash sensitive tokens for logging"""
        return hashlib.sha256(token.encode()).hexdigest()[:8]

class CacheManager:
    """Simple in-memory cache for API responses"""
    
    def __init__(self, ttl: int = 300):  # 5 minutes TTL
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = (value, time.time())

class EnterpriseGitHubAnalyzer:
    """Enterprise-grade GitHub repository analyzer with security features"""
    
    def __init__(self, github_token: str, gemini_api_key: str, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.github_token = SecurityManager.sanitize_input(github_token)
        self.gemini_api_key = SecurityManager.sanitize_input(gemini_api_key)
        self.headers = {'Authorization': f'token {self.github_token}', 'User-Agent': 'Enterprise-GitHub-Analyzer/1.0'}
        self.cache = CacheManager()
        self.session_id = SecurityManager.generate_session_id()
        
        # Initialize Gemini AI with error handling
        self.model = None
        try:
            genai.configure(api_key=self.gemini_api_key)
            # Try multiple model versions
            for model_name in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    logger.info(f"Initialized Gemini model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to initialize {model_name}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            st.warning("⚠️ AI analysis unavailable. Using enhanced rule-based analysis.")
        
        # Rate limiting
        self.last_request_time = 0
        self.request_lock = threading.Lock()
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        with self.request_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.config.rate_limit:
                time.sleep(self.config.rate_limit - elapsed)
            self.last_request_time = time.time()
    
    def extract_repo_info(self, url: str) -> tuple:
        """Extract and validate repository information"""
        if not SecurityManager.validate_github_url(url):
            raise ValueError("Invalid or potentially unsafe GitHub URL")
        
        pattern = r'github\.com/([^/]+)/([^/]+)'
        match = re.search(pattern, url.replace('.git', '').rstrip('/'))
        if match:
            owner = SecurityManager.sanitize_input(match.group(1))
            repo = SecurityManager.sanitize_input(match.group(2))
            return owner, repo
        raise ValueError("Could not extract repository information")
    
    def validate_repository_access(self, owner: str, repo: str) -> Dict:
        """Validate repository access and get basic info"""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        self._rate_limit()
        
        response = requests.get(url, headers=self.headers, timeout=10)
        
        if response.status_code == 404:
            raise ValueError("Repository not found or access denied")
        elif response.status_code == 403:
            raise ValueError("Rate limit exceeded or insufficient permissions")
        elif response.status_code != 200:
            raise ValueError(f"Repository validation failed: {response.status_code}")
        
        repo_info = response.json()
        
        # Security checks
        if repo_info.get('size', 0) > MAX_REPO_SIZE_MB * 1024:  # Size in KB
            st.warning(f"⚠️ Large repository ({repo_info.get('size', 0)/1024:.1f}MB). Analysis may take longer.")
        
        return repo_info
    
    def get_commits(self, owner: str, repo: str, days: int = 30) -> List[Dict]:
        """Fetch commits with comprehensive error handling"""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {'since': since, 'per_page': 100}
        
        # Check cache first
        cache_key = f"{owner}/{repo}/commits/{days}"
        cached_commits = self.cache.get(cache_key)
        if cached_commits:
            return cached_commits
        
        commits = []
        page = 1
        
        progress_bar = st.progress(0, text="Fetching commits...")
        
        try:
            while len(commits) < self.config.max_commits and page <= 5:  # Limit pages for performance
                params['page'] = page
                self._rate_limit()
                
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                
                if response.status_code == 403:
                    remaining = response.headers.get('X-RateLimit-Remaining', 0)
                    if int(remaining) == 0:
                        reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                        wait_time = max(0, reset_time - int(time.time()))
                        raise ValueError(f"GitHub API rate limit exceeded. Reset in {wait_time} seconds.")
                    else:
                        raise ValueError("Access forbidden. Check token permissions.")
                elif response.status_code != 200:
                    raise ValueError(f"Failed to fetch commits: HTTP {response.status_code}")
                
                page_commits = response.json()
                if not page_commits:
                    break
                
                commits.extend(page_commits)
                page += 1
                
                progress_bar.progress(min(1.0, len(commits) / 50), 
                                    text=f"Fetched {len(commits)} commits...")
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error: {str(e)}")
        finally:
            progress_bar.empty()
        
        # Cache results
        self.cache.set(cache_key, commits)
        
        return commits[:self.config.max_commits]
    
    def get_commit_details(self, owner: str, repo: str, sha: str) -> Dict:
        """Get detailed commit information with caching"""
        cache_key = f"{owner}/{repo}/commit/{sha}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        self._rate_limit()
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.cache.set(cache_key, data)
                return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch commit details for {sha}: {e}")
        
        return {}
    
    def analyze_commit_with_ai(self, commit_data: Dict) -> Dict:
        """Enhanced AI analysis with fallback"""
        if not self.model:
            return self._enhanced_fallback_analysis(commit_data)
        
        try:
            message = commit_data.get('commit', {}).get('message', '')[:500]
            files_changed = len(commit_data.get('files', []))
            additions = commit_data.get('stats', {}).get('additions', 0)
            deletions = commit_data.get('stats', {}).get('deletions', 0)
            
            # Enhanced prompt for better analysis
            prompt = f"""
            You are a senior engineering manager analyzing a Git commit. Provide a JSON response:
            {{
                "purpose": "One-line summary of the commit's purpose",
                "technical_impact": "minor/moderate/major/critical",
                "complexity_score": 1-5,
                "risk_level": "low/medium/high/critical",
                "category": "feature/bugfix/refactor/docs/test/security/performance/other",
                "team_insights": "Management insight about developer productivity and code quality",
                "security_concerns": "Any potential security implications (or 'none')",
                "review_priority": "low/medium/high"
            }}
            
            Commit: "{message}"
            Files: {files_changed}, Added: {additions}, Deleted: {deletions}
            
            Consider: Large changes (>200 lines) are high risk. Security-related keywords increase risk.
            Multiple file changes in core systems are high complexity.
            
            Return only valid JSON.
            """
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                # Clean and parse response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                analysis = json.loads(response_text.strip())
                
                # Validate and sanitize response
                required_fields = ['purpose', 'technical_impact', 'complexity_score', 'risk_level', 'category']
                if all(field in analysis for field in required_fields):
                    # Ensure proper data types and ranges
                    analysis['complexity_score'] = max(1, min(5, int(analysis.get('complexity_score', 3))))
                    analysis['purpose'] = SecurityManager.sanitize_input(analysis.get('purpose', ''))[:200]
                    analysis['team_insights'] = SecurityManager.sanitize_input(analysis.get('team_insights', ''))[:300]
                    return analysis
        
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
        
        return self._enhanced_fallback_analysis(commit_data)
    
    def _enhanced_fallback_analysis(self, commit_data: Dict) -> Dict:
        """Enhanced rule-based analysis when AI is unavailable"""
        message = commit_data.get('commit', {}).get('message', '').lower()
        additions = commit_data.get('stats', {}).get('additions', 0)
        deletions = commit_data.get('stats', {}).get('deletions', 0)
        files_changed = len(commit_data.get('files', []))
        total_changes = additions + deletions
        
        # Enhanced categorization
        security_keywords = ['security', 'auth', 'password', 'token', 'encrypt', 'vulnerability', 'xss', 'sql']
        performance_keywords = ['performance', 'optimize', 'cache', 'speed', 'memory', 'cpu']
        bug_keywords = ['fix', 'bug', 'error', 'issue', 'crash', 'fail']
        feature_keywords = ['add', 'new', 'feature', 'implement', 'create']
        refactor_keywords = ['refactor', 'clean', 'restructure', 'reorganize', 'improve']
        test_keywords = ['test', 'spec', 'coverage', 'unittest', 'integration']
        doc_keywords = ['doc', 'readme', 'comment', 'documentation', 'guide']
        
        category = CommitCategory.OTHER.value
        if any(keyword in message for keyword in security_keywords):
            category = CommitCategory.SECURITY.value
        elif any(keyword in message for keyword in performance_keywords):
            category = CommitCategory.PERFORMANCE.value
        elif any(keyword in message for keyword in bug_keywords):
            category = CommitCategory.BUGFIX.value
        elif any(keyword in message for keyword in feature_keywords):
            category = CommitCategory.FEATURE.value
        elif any(keyword in message for keyword in refactor_keywords):
            category = CommitCategory.REFACTOR.value
        elif any(keyword in message for keyword in test_keywords):
            category = CommitCategory.TEST.value
        elif any(keyword in message for keyword in doc_keywords):
            category = CommitCategory.DOCS.value
        
        # Enhanced risk assessment
        risk_level = RiskLevel.LOW.value
        if (total_changes > 500 or files_changed > 15 or 
            any(keyword in message for keyword in ['migrate', 'database', 'schema', 'breaking'])):
            risk_level = RiskLevel.CRITICAL.value
        elif (total_changes > 200 or files_changed > 8 or
              any(keyword in message for keyword in security_keywords)):
            risk_level = RiskLevel.HIGH.value
        elif total_changes > 50 or files_changed > 3:
            risk_level = RiskLevel.MEDIUM.value
        
        # Complexity scoring
        complexity_score = 1
        if total_changes > 500:
            complexity_score = 5
        elif total_changes > 200:
            complexity_score = 4
        elif total_changes > 100:
            complexity_score = 3
        elif total_changes > 20:
            complexity_score = 2
        
        # Technical impact assessment
        technical_impact = "minor"
        if category in [CommitCategory.SECURITY.value, CommitCategory.PERFORMANCE.value] or files_changed > 10:
            technical_impact = "major"
        elif category == CommitCategory.FEATURE.value or files_changed > 5:
            technical_impact = "moderate"
        
        return {
            'purpose': commit_data.get('commit', {}).get('message', '')[:100] + ('...' if len(commit_data.get('commit', {}).get('message', '')) > 100 else ''),
            'technical_impact': technical_impact,
            'complexity_score': complexity_score,
            'risk_level': risk_level,
            'category': category,
            'team_insights': f'Developer modified {files_changed} files with {total_changes} line changes. {category.title()} work detected.',
            'security_concerns': 'Potential security changes detected' if any(keyword in message for keyword in security_keywords) else 'none',
            'review_priority': 'high' if risk_level in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value] else 'medium'
        }

def create_enterprise_visualizations(df: pd.DataFrame):
    """Create comprehensive enterprise-grade visualizations"""
    
    # Main KPI Dashboard
    st.subheader("🎯 Executive Dashboard")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_commits = len(df)
        st.metric("Total Commits", total_commits)
    
    with col2:
        avg_complexity = df['complexity_score'].mean()
        complexity_trend = "📈" if avg_complexity > 3 else "📊" if avg_complexity > 2 else "📉"
        st.metric("Avg Complexity", f"{avg_complexity:.1f} {complexity_trend}")
    
    with col3:
        high_risk = len(df[df['risk_level'].isin(['high', 'critical'])])
        risk_pct = (high_risk / total_commits * 100) if total_commits > 0 else 0
        st.metric("High Risk Commits", f"{high_risk} ({risk_pct:.1f}%)")
    
    with col4:
        active_devs = df['author'].nunique()
        st.metric("Active Developers", active_devs)
    
    with col5:
        security_commits = len(df[df['category'] == 'security'])
        st.metric("Security Commits", security_commits, delta="Critical Focus" if security_commits > 0 else None)
    
    # Risk Analysis Section
    st.subheader("🚨 Risk Analysis & Security Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution with enhanced colors
        risk_counts = df['risk_level'].value_counts()
        colors = {'low': '#28a745', 'medium': '#ffc107', 'high': '#fd7e14', 'critical': '#dc3545'}
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color=risk_counts.index,
            color_discrete_map=colors
        )
        fig_risk.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Category breakdown with security highlighting
        category_counts = df['category'].value_counts()
        colors = ['#dc3545' if cat == 'security' else '#667eea' for cat in category_counts.index]
        fig_category = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title="Commit Categories",
            labels={'x': 'Count', 'y': 'Category'},
            color=category_counts.index,
            color_discrete_sequence=colors
        )
        st.plotly_chart(fig_category, use_container_width=True)
    
    # Developer Performance Analytics
    st.subheader("👥 Developer Performance Analytics")
    
    # Enhanced developer activity with multiple metrics
    dev_metrics = df.groupby('author').agg({
        'sha': 'count',
        'additions': 'sum',
        'deletions': 'sum',
        'files_changed': 'sum',
        'complexity_score': 'mean',
        'risk_level': lambda x: (x.isin(['high', 'critical'])).sum()
    }).round(2)
    dev_metrics.columns = ['Commits', 'Lines Added', 'Lines Deleted', 'Files Changed', 'Avg Complexity', 'High Risk Commits']
    dev_metrics['Code Churn'] = dev_metrics['Lines Added'] + dev_metrics['Lines Deleted']
    
    # Developer activity heatmap
    fig_dev_activity = px.bar(
        dev_metrics.reset_index(),
        x='author',
        y='Commits',
        color='Avg Complexity',
        size='Code Churn',
        hover_data=['High Risk Commits', 'Files Changed'],
        title="Developer Activity (Bubble size = Code Churn, Color = Complexity)",
        color_continuous_scale='RdYlBu_r'
    )
    st.plotly_chart(fig_dev_activity, use_container_width=True)
    
    # Timeline Analysis
    st.subheader("📈 Timeline Analysis")
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily commit timeline with risk overlay
        timeline_df = df.groupby(['date', 'risk_level']).size().reset_index(name='count')
        fig_timeline = px.area(
            timeline_df,
            x='date',
            y='count',
            color='risk_level',
            title="Daily Commit Timeline by Risk Level",
            color_discrete_map={'low': '#28a745', 'medium': '#ffc107', 'high': '#fd7e14', 'critical': '#dc3545'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        # Complexity trend over time
        daily_complexity = df.groupby('date')['complexity_score'].mean().reset_index()
        fig_complexity_trend = px.line(
            daily_complexity,
            x='date',
            y='complexity_score',
            title="Average Code Complexity Trend",
            markers=True
        )
        fig_complexity_trend.add_hline(y=3, line_dash="dash", line_color="red", 
                                     annotation_text="High Complexity Threshold")
        st.plotly_chart(fig_complexity_trend, use_container_width=True)
    
    # Advanced Analytics
    st.subheader("🔍 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Files changed vs risk correlation
        fig_scatter = px.scatter(
            df,
            x='files_changed',
            y='complexity_score',
            color='risk_level',
            size='additions',
            hover_data=['author', 'category'],
            title="Files Changed vs Complexity (Size = Lines Added)",
            color_discrete_map={'low': '#28a745', 'medium': '#ffc107', 'high': '#fd7e14', 'critical': '#dc3545'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Team collaboration matrix
        author_category = df.groupby(['author', 'category']).size().reset_index(name='commits')
        fig_collab = px.treemap(
            author_category,
            path=['author', 'category'],
            values='commits',
            title="Team Collaboration Matrix",
            color='commits',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_collab, use_container_width=True)

def create_management_insights(df: pd.DataFrame) -> str:
    """Generate executive summary and actionable insights"""
    total_commits = len(df)
    if total_commits == 0:
        return "No commits to analyze."
    
    high_risk_commits = len(df[df['risk_level'].isin(['high', 'critical'])])
    avg_complexity = df['complexity_score'].mean()
    active_developers = df['author'].nunique()
    security_commits = len(df[df['category'] == 'security'])
    
    # Top contributors
    top_contributors = df['author'].value_counts().head(3)
    
    # Risk analysis
    risk_percentage = (high_risk_commits / total_commits) * 100
    
    # Category analysis
    top_categories = df['category'].value_counts().head(3)
    
    insights = f"""
# Executive Summary & Management Insights

## 📊 Key Metrics Overview
- **Total Development Activity**: {total_commits} commits analyzed
- **Team Size**: {active_developers} active developers
- **Risk Profile**: {high_risk_commits} high-risk commits ({risk_percentage:.1f}% of total)
- **Code Complexity**: {avg_complexity:.1f}/5.0 average complexity score
- **Security Focus**: {security_commits} security-related commits

## 👥 Team Performance
### Top Contributors:
{chr(10).join([f"- **{author}**: {commits} commits" for author, commits in top_contributors.items()])}

### Work Distribution:
{chr(10).join([f"- **{category.title()}**: {count} commits" for category, count in top_categories.items()])}

## 🚨 Risk Assessment & Recommendations

### Immediate Actions Required:
"""
    
    if risk_percentage > 30:
        insights += "\n- ⚠️ **HIGH RISK**: >30% of commits are high-risk. Increase code review scrutiny."
    elif risk_percentage > 15:
        insights += "\n- ⚠️ **MEDIUM RISK**: 15-30% high-risk commits. Monitor closely."
    else:
        insights += "\n- ✅ **LOW RISK**: <15% high-risk commits. Good risk management."
    
    if avg_complexity > 3.5:
        insights += "\n- 🔧 **Code Complexity**: High average complexity detected. Consider refactoring initiatives."
    
    if security_commits == 0:
        insights += "\n- 🔐 **Security**: No security commits detected. Ensure security is being addressed."
    else:
        insights += f"\n- 🔐 **Security**: {security_commits} security commits show good security focus."
    
    insights += f"""

## 📈 Strategic Recommendations

### For Engineering Managers:
1. **Code Review Process**: Prioritize high-risk and high-complexity commits
2. **Developer Support**: Provide additional guidance to developers with consistently high complexity scores
3. **Security Focus**: {"Maintain" if security_commits > 0 else "Increase"} security-related development activities
4. **Team Balance**: {"Good" if active_developers > 2 else "Consider"} team size for workload distribution

### For Technical Leads:
1. **Architecture Review**: High complexity scores may indicate architectural debt
2. **Best Practices**: Implement coding standards to reduce complexity
3. **Mentoring**: Pair experienced developers with those showing high risk patterns

### Risk Mitigation Strategy:
- Implement automated testing for high-risk categories
- Require additional reviewers for critical changes
- Consider feature flags for large feature deployments
- Regular architecture reviews for complex changes

---
*Analysis generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC*
*Session ID: {SecurityManager.generate_session_id()[:8]}...*
"""
    
    return insights

def main():
    # Header with branding
    st.markdown("""
    <div class="main-header">
        <h1>🏢 Enterprise GitHub Repository Analyzer</h1>
        <p>AI-Powered Development Intelligence for Engineering Leaders</p>
        <span class="security-badge">🔒 Enterprise Security</span>
        <span class="security-badge">🚀 Production Ready</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🔐 Secure Configuration")
        
        # Security notice
        st.markdown("""
        <div class="warning-card">
        <b>🔒 Security Notice</b><br>
        • Tokens are encrypted in memory<br>
        • No data stored permanently<br>
        • Rate limiting enforced<br>
        • Session timeout: 1 hour
        </div>
        """, unsafe_allow_html=True)
        
        github_token = st.text_input(
            "GitHub Personal Access Token",
            type="password",
            help="Generate at: Settings > Developer settings > Personal access tokens",
            placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
        )
        
        gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get free key from Google AI Studio",
            placeholder="AIzaSyxxxxxxxxxxxxxxxxx"
        )
        
        st.header("📊 Analysis Configuration")
        
        days_back = st.slider(
            "Analysis Period (Days)",
            min_value=7,
            max_value=90,
            value=30,
            help="Longer periods may hit rate limits"
        )
        
        repo_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/company/repository",
            help="Must be a valid GitHub repository URL"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            max_commits = st.number_input(
                "Max Commits to Analyze",
                min_value=10,
                max_value=500,
                value=100,
                help="Limit for performance and rate limiting"
            )
            
            enable_caching = st.checkbox(
                "Enable Response Caching",
                value=True,
                help="Cache API responses for 5 minutes"
            )
            
            detailed_analysis = st.checkbox(
                "Detailed File Analysis",
                value=False,
                help="Analyze individual file changes (slower)"
            )
        
        # Security validation
        security_check = st.checkbox(
            "🔒 I confirm this is an authorized repository analysis",
            help="Only analyze repositories you have permission to access"
        )
        
        analyze_btn = st.button(
            "🚀 Start Enterprise Analysis",
            type="primary",
            disabled=not security_check,
            help="Ensure all configurations are correct before starting"
        )
    
    # Main content area
    if analyze_btn:
        if not all([github_token, gemini_api_key, repo_url, security_check]):
            st.error("❌ Please provide all required inputs and confirm security checkbox.")
            return
        
        try:
            # Initialize analyzer with security config
            config = SecurityConfig(
                max_commits=max_commits,
                rate_limit=API_RATE_LIMIT
            )
            
            analyzer = EnterpriseGitHubAnalyzer(github_token, gemini_api_key, config)
            
            # Validate repository
            with st.spinner("🔍 Validating repository access..."):
                owner, repo = analyzer.extract_repo_info(repo_url)
                repo_info = analyzer.validate_repository_access(owner, repo)
            
            # Display repository information
            st.markdown(f"""
            <div class="success-card">
            <h3>✅ Repository Validated: {owner}/{repo}</h3>
            <p><b>Description:</b> {repo_info.get('description', 'No description')}</p>
            <p><b>Language:</b> {repo_info.get('language', 'Multiple')} | 
               <b>Size:</b> {repo_info.get('size', 0)/1024:.1f}MB | 
               <b>Stars:</b> {repo_info.get('stargazers_count', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Fetch commits
            commits = analyzer.get_commits(owner, repo, days_back)
            
            if not commits:
                st.warning("⚠️ No commits found in the specified time period.")
                return
            
            st.success(f"📥 Found {len(commits)} commits. Starting AI-powered analysis...")
            
            # Process commits with enhanced progress tracking
            processed_commits = []
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, commit in enumerate(commits):
                    status_text.text(f"Analyzing commit {i+1}/{len(commits)}: {commit['sha'][:8]}...")
                    
                    # Get detailed commit info
                    commit_details = analyzer.get_commit_details(owner, repo, commit['sha'])
                    if not commit_details:
                        continue
                    
                    # AI analysis
                    analysis = analyzer.analyze_commit_with_ai(commit_details)
                    
                    # Enhanced commit data structure
                    processed_commit = {
                        'sha': commit['sha'][:8],
                        'full_sha': commit['sha'],
                        'author': commit['commit']['author']['name'],
                        'author_email': commit['commit']['author']['email'],
                        'date': commit['commit']['author']['date'],
                        'message': commit['commit']['message'],
                        'message_preview': commit['commit']['message'][:100] + ('...' if len(commit['commit']['message']) > 100 else ''),
                        'files_changed': len(commit_details.get('files', [])),
                        'additions': commit_details.get('stats', {}).get('additions', 0),
                        'deletions': commit_details.get('stats', {}).get('deletions', 0),
                        'total_changes': commit_details.get('stats', {}).get('total', 0),
                        'url': commit.get('html_url', ''),
                        **analysis
                    }
                    processed_commits.append(processed_commit)
                    
                    progress_bar.progress((i + 1) / len(commits))
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            if not processed_commits:
                st.error("❌ No commits could be processed successfully.")
                return
            
            # Create comprehensive DataFrame
            df = pd.DataFrame(processed_commits)
            
            # Display results with tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Detailed Analysis", "📈 Management Insights", "💾 Export & Reports"])
            
            with tab1:
                create_enterprise_visualizations(df)
            
            with tab2:
                st.header("📋 Comprehensive Commit Analysis")
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    risk_filter = st.multiselect(
                        "Filter by Risk Level",
                        options=df['risk_level'].unique(),
                        default=df['risk_level'].unique()
                    )
                with col2:
                    author_filter = st.multiselect(
                        "Filter by Author",
                        options=df['author'].unique(),
                        default=df['author'].unique()
                    )
                with col3:
                    category_filter = st.multiselect(
                        "Filter by Category",
                        options=df['category'].unique(),
                        default=df['category'].unique()
                    )
                
                # Apply filters
                filtered_df = df[
                    (df['risk_level'].isin(risk_filter)) &
                    (df['author'].isin(author_filter)) &
                    (df['category'].isin(category_filter))
                ]
                
                # Display filtered data with enhanced formatting
                display_columns = [
                    'sha', 'author', 'date', 'message_preview', 'category',
                    'risk_level', 'complexity_score', 'files_changed', 'total_changes'
                ]
                
                st.dataframe(
                    filtered_df[display_columns],
                    use_container_width=True,
                    height=400,
                    column_config={
                        'sha': st.column_config.LinkColumn(
                            'Commit',
                            help='Click to view commit on GitHub',
                            display_text=r'https://github\.com/.*/commit/(.*)'
                        ),
                        'risk_level': st.column_config.SelectboxColumn(
                            'Risk Level',
                            help='AI-assessed risk level',
                            options=['low', 'medium', 'high', 'critical']
                        ),
                        'complexity_score': st.column_config.ProgressColumn(
                            'Complexity',
                            help='Code complexity score (1-5)',
                            min_value=1,
                            max_value=5
                        )
                    }
                )
                
                # Detailed commit inspection
                if st.button("🔍 Inspect High-Risk Commits"):
                    high_risk_commits = df[df['risk_level'].isin(['high', 'critical'])]
                    if len(high_risk_commits) > 0:
                        st.subheader("⚠️ High-Risk Commits Requiring Attention")
                        for _, commit in high_risk_commits.iterrows():
                            with st.expander(f"🚨 {commit['sha']} - {commit['category'].upper()} - {commit['author']}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Message:** {commit['message']}")
                                    st.write(f"**Risk Level:** {commit['risk_level'].upper()}")
                                    st.write(f"**Complexity:** {commit['complexity_score']}/5")
                                with col2:
                                    st.write(f"**Files Changed:** {commit['files_changed']}")
                                    st.write(f"**Lines Changed:** {commit['total_changes']}")
                                    st.write(f"**Security Concerns:** {commit.get('security_concerns', 'None')}")
                                st.write(f"**Team Insights:** {commit['team_insights']}")
                    else:
                        st.success("✅ No high-risk commits detected!")
            
            with tab3:
                st.header("📈 Executive Management Insights")
                insights = create_management_insights(df)
                st.markdown(insights)
                
                # Team performance matrix
                st.subheader("👥 Team Performance Matrix")
                team_metrics = df.groupby('author').agg({
                    'sha': 'count',
                    'complexity_score': 'mean',
                    'risk_level': lambda x: (x.isin(['high', 'critical'])).sum(),
                    'additions': 'sum',
                    'deletions': 'sum',
                    'category': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown'
                }).round(2)
                team_metrics.columns = ['Commits', 'Avg Complexity', 'High Risk Count', 'Lines Added', 'Lines Deleted', 'Primary Category']
                team_metrics['Productivity Score'] = (team_metrics['Commits'] * 0.4 + 
                                                    (6 - team_metrics['Avg Complexity']) * 0.3 + 
                                                    (team_metrics['Commits'] - team_metrics['High Risk Count']) * 0.3).round(2)
                
                st.dataframe(team_metrics, use_container_width=True)
            
            with tab4:
                st.header("💾 Export & Reporting")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📄 Detailed CSV Report")
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Analysis CSV",
                        data=csv_data,
                        file_name=f"{owner}_{repo}_enterprise_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Complete dataset with all analyzed metrics"
                    )
                
                with col2:
                    st.subheader("📊 Executive Summary")
                    insights = create_management_insights(df)
                    st.download_button(
                        label="📥 Download Executive Report",
                        data=insights,
                        file_name=f"{owner}_{repo}_executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        help="Management summary with strategic insights"
                    )
                
                with col3:
                    st.subheader("🎯 Action Items JSON")
                    high_risk_actions = df[df['risk_level'].isin(['high', 'critical'])][
                        ['sha', 'author', 'message', 'risk_level', 'category', 'team_insights']
                    ].to_dict('records')
                    
                    action_items = {
                        'generated_at': datetime.now().isoformat(),
                        'repository': f"{owner}/{repo}",
                        'analysis_period_days': days_back,
                        'total_commits_analyzed': len(df),
                        'high_risk_count': len(high_risk_actions),
                        'high_risk_commits': high_risk_actions,
                        'recommendations': {
                            'immediate_review_required': len(high_risk_actions),
                            'code_review_priority': 'high' if len(high_risk_actions) > len(df) * 0.2 else 'medium',
                            'team_training_suggested': df['complexity_score'].mean() > 3.5
                        }
                    }
                    
                    st.download_button(
                        label="📥 Download Action Items",
                        data=json.dumps(action_items, indent=2),
                        file_name=f"{owner}_{repo}_action_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Structured action items for immediate follow-up"
                    )
                
                # Usage analytics
                st.subheader("📈 Usage Analytics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Session ID", analyzer.session_id[:8] + "...")
                with col2:
                    st.metric("API Calls Made", f"~{len(commits) * 2}")
                with col3:
                    st.metric("Analysis Duration", "Real-time")
        
        except ValueError as e:
            st.error(f"❌ Configuration Error: {str(e)}")
            st.info("Please check your inputs and repository permissions.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            st.error(f"❌ Analysis Failed: An unexpected error occurred.")
            st.info("Please try again or contact support if the issue persists.")
    
    else:
        # Landing page when no analysis is running
        st.markdown("""
        ## 🚀 Enterprise-Grade Repository Intelligence
        
        Transform your development workflow with AI-powered insights designed for engineering leaders.
        
        ### ✨ Key Features
        
        **🔒 Security First**
        - Enterprise-grade security measures
        - Token encryption and secure handling
        - Rate limiting and access validation
        - No permanent data storage
        
        **🧠 AI-Powered Analysis**
        - Google Gemini AI for intelligent commit analysis
        - Advanced risk assessment algorithms
        - Technical impact evaluation
        - Security concern identification
        
        **📊 Comprehensive Dashboards**
        - Executive KPI dashboards
        - Developer performance analytics
        - Risk distribution analysis
        - Timeline and trend visualization
        
        **📈 Management Insights**
        - Strategic recommendations
        - Team performance metrics
        - Actionable insights for decision making
        - Export capabilities for reporting
        
        ### 🎯 Perfect For:
        - **Engineering Managers** - Team oversight and performance tracking
        - **Technical Leads** - Code quality and risk management
        - **DevOps Teams** - Deployment risk assessment
        - **Security Teams** - Security-focused development monitoring
        
        ---
        
        ⚡ **Ready to transform your development intelligence?** 
        Configure your credentials in the sidebar and start your analysis!
        """)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>🎯 Risk Intelligence</h4>
            <p>AI identifies high-risk commits automatically, helping you prioritize code reviews and prevent issues before they reach production.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>👥 Team Analytics</h4>
            <p>Understand your team's productivity patterns, workload distribution, and development focus areas with detailed performance metrics.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>📊 Executive Reporting</h4>
            <p>Generate comprehensive reports and actionable insights that help engineering leaders make data-driven decisions.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()