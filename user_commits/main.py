import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="GitHub Commit Fetcher", layout="wide")

st.title("🔍 GitHub Commit Fetcher")
st.markdown("Fetch all commits made by a user across all repositories.")

# Input fields
username = st.text_input("Enter GitHub username")
token = st.text_input("Enter Personal Access Token (optional)", type="password")

if st.button("Fetch Commits"):
    if not username.strip():
        st.error("Please enter a GitHub username.")
    else:
        headers = {"Authorization": f"token {token}"} if token else {}
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"
        
        with st.spinner("Fetching repositories..."):
            repos_response = requests.get(repos_url, headers=headers)
            if repos_response.status_code != 200:
                st.error(f"Error fetching repos: {repos_response.status_code}")
                st.stop()
            repos = repos_response.json()

        all_commits = []
        with st.spinner("Fetching commits..."):
            for repo in repos:
                repo_name = repo['name']
                owner = repo['owner']['login']
                commits_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits?author={username}&per_page=100"
                commits_response = requests.get(commits_url, headers=headers)
                if commits_response.status_code == 200:
                    commits = commits_response.json()
                    for commit in commits:
                        all_commits.append({
                            "Repository": repo_name,
                            "Message": commit['commit']['message'],
                            "Date": commit['commit']['author']['date'],
                            "URL": commit['html_url']
                        })

        if all_commits:
            df = pd.DataFrame(all_commits)
            st.success(f"Found {len(df)} commits in {len(repos)} repositories.")
            st.dataframe(df, use_container_width=True)

            # CSV download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"{username}_commits.csv",
                mime="text/csv"
            )
        else:
            st.warning("No commits found for this user.")

st.markdown("---")
st.markdown("💡 **Tip:** Use a personal access token for higher rate limits (5000 requests/hour vs 60 without).")
