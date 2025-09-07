import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import re
from typing import Dict, List, Any
import google.generativeai as genai
from io import StringIO

# Configure page
st.set_page_config(
    page_title="GitHub Repository Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GitHubAnalyzer:
    def __init__(self, github_token: str, gemini_api_key: str):
        self.github_token = github_token
        self.headers = {'Authorization': f'token {github_token}'}
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.rate_limit_delay = 1  # seconds between API calls
        
    def extract_repo_info(self, url: str) -> tuple:
        """Extract owner and repo name from GitHub URL"""
        pattern = r'github\.com/([^/]+)/([^/]+)'
        match = re.search(pattern, url.replace('.git', ''))
        if match:
            return match.group(1), match.group(2)
        raise ValueError("Invalid GitHub URL format")
    
    def get_commits(self, owner: str, repo: str, days: int = 30) -> List[Dict]:
        """Fetch commits from last N days"""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {'since': since, 'per_page': 100}
        
        commits = []
        page = 1
        
        with st.spinner(f"Fetching commits from last {days} days..."):
            while len(commits) < 500:  # Limit for performance
                params['page'] = page
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 403:
                    st.error("Rate limit exceeded. Please try again later.")
                    break
                elif response.status_code != 200:
                    st.error(f"Error fetching commits: {response.status_code}")
                    break
                
                page_commits = response.json()
                if not page_commits:
                    break
                    
                commits.extend(page_commits)
                page += 1
                time.sleep(self.rate_limit_delay)
        
        return commits
    
    def get_commit_details(self, owner: str, repo: str, sha: str) -> Dict:
        """Get detailed commit information including diff"""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        response = requests.get(url, headers=self.headers)
        time.sleep(self.rate_limit_delay)
        
        if response.status_code == 200:
            return response.json()
        return {}
    
    def analyze_commit_with_ai(self, commit_data: Dict) -> Dict:
        """Analyze commit using Gemini AI"""
        try:
            # Prepare commit info for analysis
            message = commit_data.get('commit', {}).get('message', '')
            files_changed = len(commit_data.get('files', []))
            additions = commit_data.get('stats', {}).get('additions', 0)
            deletions = commit_data.get('stats', {}).get('deletions', 0)
            
            # Create analysis prompt
            prompt = f"""
            Analyze this Git commit and provide a JSON response with the following structure:
            {{
                "purpose": "Brief summary of what this commit does",
                "technical_impact": "Assessment of technical impact (minor/moderate/major)",
                "complexity_score": 1-5 (integer),
                "risk_level": "low/medium/high",
                "category": "feature/bugfix/refactor/docs/test/other",
                "team_insights": "Management insights about this change"
            }}
            
            Commit Message: {message}
            Files Changed: {files_changed}
            Lines Added: {additions}
            Lines Deleted: {deletions}
            
            Provide only valid JSON response.
            """
            
            response = self.model.generate_content(prompt)
            
            # Parse AI response
            try:
                analysis = json.loads(response.text.strip())
                return analysis
            except json.JSONDecodeError:
                # Fallback analysis if AI response isn't valid JSON
                return self._fallback_analysis(commit_data)
                
        except Exception as e:
            st.warning(f"AI analysis failed: {str(e)}. Using fallback analysis.")
            return self._fallback_analysis(commit_data)
    
    def _fallback_analysis(self, commit_data: Dict) -> Dict:
        """Fallback analysis when AI fails"""
        message = commit_data.get('commit', {}).get('message', '').lower()
        additions = commit_data.get('stats', {}).get('additions', 0)
        deletions = commit_data.get('stats', {}).get('deletions', 0)
        files_changed = len(commit_data.get('files', []))
        
        # Simple rule-based analysis
        if any(word in message for word in ['fix', 'bug', 'error']):
            category = 'bugfix'
        elif any(word in message for word in ['add', 'new', 'feature']):
            category = 'feature'
        elif any(word in message for word in ['refactor', 'clean', 'improve']):
            category = 'refactor'
        elif any(word in message for word in ['doc', 'readme', 'comment']):
            category = 'docs'
        else:
            category = 'other'
        
        total_changes = additions + deletions
        complexity_score = min(5, max(1, (total_changes // 50) + 1))
        
        if total_changes > 200 or files_changed > 10:
            risk_level = 'high'
        elif total_changes > 50 or files_changed > 3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'purpose': commit_data.get('commit', {}).get('message', '')[:100] + '...',
            'technical_impact': 'moderate',
            'complexity_score': complexity_score,
            'risk_level': risk_level,
            'category': category,
            'team_insights': f'Developer made {files_changed} file changes with {total_changes} line modifications'
        }

def create_visualizations(df: pd.DataFrame):
    """Create all visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Developer Activity")
        activity_df = df.groupby('author').agg({
            'sha': 'count',
            'additions': 'sum',
            'deletions': 'sum',
            'complexity_score': 'mean'
        }).round(2)
        activity_df.columns = ['Commits', 'Additions', 'Deletions', 'Avg Complexity']
        
        fig_activity = px.bar(
            activity_df.reset_index(), 
            x='author', 
            y='Commits',
            title="Commits per Developer",
            color='Avg Complexity',
            color_continuous_scale='RdYlBu_r'
        )
        st.plotly_chart(fig_activity, use_container_width=True)
    
    with col2:
        st.subheader("Risk Distribution")
        risk_counts = df['risk_level'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            title="Risk Level Distribution",
            color_discrete_map={'low': 'green', 'medium': 'orange', 'high': 'red'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    st.subheader("Commit Timeline")
    df['date'] = pd.to_datetime(df['date']).dt.date
    timeline_df = df.groupby(['date', 'risk_level']).size().reset_index(name='count')
    
    fig_timeline = px.line(
        timeline_df, 
        x='date', 
        y='count',
        color='risk_level',
        title="Commits Over Time by Risk Level",
        color_discrete_map={'low': 'green', 'medium': 'orange', 'high': 'red'}
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Category Breakdown")
        category_counts = df['category'].value_counts()
        fig_category = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Commit Categories",
            labels={'x': 'Category', 'y': 'Count'}
        )
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col4:
        st.subheader("Productivity Metrics")
        metrics_df = df.groupby('author').agg({
            'additions': 'sum',
            'deletions': 'sum',
            'files_changed': 'sum'
        }).reset_index()
        
        fig_productivity = go.Figure()
        fig_productivity.add_trace(go.Scatter(
            x=metrics_df['additions'],
            y=metrics_df['deletions'],
            mode='markers+text',
            text=metrics_df['author'],
            textposition='top center',
            marker=dict(
                size=metrics_df['files_changed'] * 2,
                color=metrics_df['files_changed'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig_productivity.update_layout(
            title="Developer Productivity (Bubble size = Files changed)",
            xaxis_title="Lines Added",
            yaxis_title="Lines Deleted"
        )
        st.plotly_chart(fig_productivity, use_container_width=True)

def main():
    st.title("🔍 GitHub Repository Analyzer for Team Leads")
    st.markdown("Analyze repository commits with AI-powered insights for better team management.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        github_token = st.text_input("GitHub Token", type="password", help="Personal Access Token for GitHub API")
        gemini_api_key = st.text_input("Gemini API Key", type="password", help="Google Gemini API key")
        
        if gemini_api_key:
            st.info("💡 If AI analysis fails, try getting a new API key from Google AI Studio")
        
        st.header("Analysis Settings")
        days_back = st.slider("Days to analyze", min_value=7, max_value=90, value=30)
        repo_url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo")
        
        analyze_btn = st.button("🚀 Start Analysis", type="primary")
    
    if analyze_btn:
        if not all([github_token, gemini_api_key, repo_url]):
            st.error("Please provide all required inputs.")
            return
        
        try:
            analyzer = GitHubAnalyzer(github_token, gemini_api_key)
            owner, repo = analyzer.extract_repo_info(repo_url)
            
            st.info(f"Analyzing repository: {owner}/{repo}")
            
            # Fetch commits
            commits = analyzer.get_commits(owner, repo, days_back)
            
            if not commits:
                st.warning("No commits found in the specified time period.")
                return
            
            st.success(f"Found {len(commits)} commits. Starting AI analysis...")
            
            # Process commits with progress bar
            processed_commits = []
            progress_bar = st.progress(0)
            
            for i, commit in enumerate(commits):
                # Get detailed commit info
                commit_details = analyzer.get_commit_details(owner, repo, commit['sha'])
                if not commit_details:
                    continue
                
                # AI analysis
                analysis = analyzer.analyze_commit_with_ai(commit_details)
                
                # Combine data
                processed_commit = {
                    'sha': commit['sha'][:8],
                    'author': commit['commit']['author']['name'],
                    'date': commit['commit']['author']['date'],
                    'message': commit['commit']['message'][:100] + '...',
                    'files_changed': len(commit_details.get('files', [])),
                    'additions': commit_details.get('stats', {}).get('additions', 0),
                    'deletions': commit_details.get('stats', {}).get('deletions', 0),
                    **analysis
                }
                processed_commits.append(processed_commit)
                
                progress_bar.progress((i + 1) / len(commits))
                
                # Limit processing for demo
                if i >= 49:  # Process max 50 commits
                    break
            
            if not processed_commits:
                st.error("No commits could be processed.")
                return
            
            # Create DataFrame
            df = pd.DataFrame(processed_commits)
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Commits", len(df))
            with col2:
                st.metric("Avg Complexity", f"{df['complexity_score'].mean():.1f}")
            with col3:
                st.metric("High Risk Commits", len(df[df['risk_level'] == 'high']))
            with col4:
                st.metric("Active Developers", df['author'].nunique())
            
            # Visualizations
            create_visualizations(df)
            
            # Detailed data table
            st.header("📋 Detailed Commit Analysis")
            st.dataframe(df, use_container_width=True)
            
            # Export functionality
            st.header("💾 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name=f"{owner}_{repo}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Team insights summary
                insights = df.groupby('author').agg({
                    'complexity_score': 'mean',
                    'risk_level': lambda x: (x == 'high').sum(),
                    'category': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown'
                }).round(2)
                
                summary = f"""
                # Team Analysis Summary for {owner}/{repo}
                
                ## Key Findings:
                - **Total Commits Analyzed**: {len(df)}
                - **Analysis Period**: Last {days_back} days
                - **Active Developers**: {df['author'].nunique()}
                - **High Risk Commits**: {len(df[df['risk_level'] == 'high'])} ({len(df[df['risk_level'] == 'high'])/len(df)*100:.1f}%)
                
                ## Developer Insights:
                {insights.to_string()}
                
                ## Recommendations:
                - Focus on commits with high risk levels for code review priority
                - Developers with high complexity scores may need additional support
                - Monitor commit patterns for workload distribution
                """
                
                st.download_button(
                    label="Download Summary Report",
                    data=summary,
                    file_name=f"{owner}_{repo}_summary_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.info("Please check your inputs and try again.")

if __name__ == "__main__":
    main()