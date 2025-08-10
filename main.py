import requests
import streamlit as st
from datetime import datetime
import csv
import os
import pandas as pd

# GitHub API base
GITHUB_API = "https://api.github.com"

st.title("LeadMate - AI Code Strategist")

# CSV file configuration
CSV_FILE = "commit_database.csv"
CSV_HEADERS = ["Repository", "Commit_SHA", "Commit_Message", "Author", "Date", "Timestamp"]

# Add control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Database"):
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
            st.success("✅ Database cleared successfully!")
        else:
            st.info("ℹ️ No database file found to clear")
with col2:
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            st.metric("📊 Total Commits", len(df))
        except:
            st.metric("📊 Total Commits", "Error")
    else:
        st.metric("📊 Total Commits", "0")

repo_url = st.text_input("Enter GitHub Repository URL (e.g., https://github.com/user/repo)")



def commit_exists(repo_name, commit_sha):
    """Check if a commit already exists in the CSV file"""
    if not os.path.exists(CSV_FILE):
        return False

    try:
        df = pd.read_csv(CSV_FILE)
        # Check if combination of repository and commit SHA already exists
        exists = ((df['Repository'] == repo_name) & (df['Commit_SHA'] == commit_sha)).any()
        return exists
    except Exception:
        return False

def save_to_csv(repo_name, commit_data):
    """Save commit data to CSV file if it doesn't already exist"""
    # Check if commit already exists
    if commit_exists(repo_name, commit_data['sha']):
        return False  # Commit already exists, don't add

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(CSV_HEADERS)

        # Write commit data
        writer.writerow([
            repo_name,
            commit_data['sha'],
            commit_data['message'],
            commit_data['author'],
            commit_data['date'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])

    return True  # Commit was added

def get_commits(owner, repo):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raises an HTTPError for bad responses
        return r.json()[:5]  # latest 5 commits
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch commits: {e}")
        return []

def analyze_code(diff_text):
    # Mock AI response for demo (replace with actual LLM call)
    # Using diff_text to make analysis more realistic
    if not diff_text:
        return {
            "summary": "No code changes detected in this commit.",
            "suggestion": "This might be a merge commit or documentation update."
        }

    # Simple analysis based on diff content
    if "test" in diff_text.lower():
        return {
            "summary": "Test-related changes detected in the codebase.",
            "suggestion": "Good practice! Continue maintaining comprehensive test coverage."
        }
    elif "fix" in diff_text.lower() or "bug" in diff_text.lower():
        return {
            "summary": "Bug fix or error correction identified.",
            "suggestion": "Consider adding regression tests to prevent similar issues."
        }
    else:
        return {
            "summary": "Code changes detected - appears to be feature development or refactoring.",
            "suggestion": "Ensure proper documentation and testing for new functionality."
        }

if repo_url:
    try:
        # Validate and parse GitHub URL
        repo_url = repo_url.strip()
        if not repo_url.startswith("https://github.com/"):
            st.error("Please enter a valid GitHub repository URL (e.g., https://github.com/user/repo)")
        else:
            parts = repo_url.split("/")
            if len(parts) < 5:
                st.error("Invalid GitHub URL format. Please use: https://github.com/owner/repository")
            else:
                owner, repo = parts[-2], parts[-1]
                # Remove .git suffix if present
                if repo.endswith('.git'):
                    repo = repo[:-4]

                commits = get_commits(owner, repo)

                if not commits:
                    st.warning("No commits found or unable to fetch repository data.")
                else:
                    st.success(f"Found {len(commits)} recent commits")

                    # Show progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Track statistics
                    added_count = 0
                    skipped_count = 0

                    for i, commit in enumerate(commits):
                        try:
                            # Update progress
                            progress = (i + 1) / len(commits)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing commit {i + 1} of {len(commits)}")

                            sha = commit['sha']
                            message = commit['commit']['message']
                            author = commit['commit']['author']['name']
                            date = commit['commit']['author']['date']



                            # Display commit information
                            st.subheader(f"Commit: {sha[:7]}")
                            st.markdown(f"**Commit Message:** {message}")
                            formatted_date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
                            st.write(f"**Author:** {author} | **Date:** {formatted_date}")
                        


                            # Save to CSV
                            commit_data = {
                                'sha': sha,
                                'message': message,
                                'author': author,
                                'date': formatted_date
                            }

                            # Try to save and track if it was added or skipped
                            was_added = save_to_csv(f"{owner}/{repo}", commit_data)
                            if was_added:
                                added_count += 1
                                st.success(f"✅ Added to database")
                            else:
                                skipped_count += 1
                                st.info(f"ℹ️ Already exists in database - skipped")

                            st.markdown("---")

                        except KeyError as e:
                            st.error(f"Error processing commit {sha[:7] if 'sha' in locals() else 'unknown'}: Missing data field {e}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error fetching commit details: {e}")
                        except Exception as e:
                            st.error(f"Unexpected error processing commit: {e}")

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Show completion message and CSV info
                    st.success(f"✅ Successfully processed {len(commits)} commits!")

                    # Show statistics
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("📝 New Commits Added", added_count)
                    with col_stat2:
                        st.metric("⏭️ Duplicates Skipped", skipped_count)

                    st.info(f"📁 Data saved to: {CSV_FILE}")

                    # Display CSV data if it exists
                    if os.path.exists(CSV_FILE):
                        st.subheader("📊 Database Preview")
                        try:
                            df = pd.read_csv(CSV_FILE)
                            st.dataframe(df.tail(10))  # Show last 10 entries

                            # Download button
                            with open(CSV_FILE, 'rb') as file:
                                st.download_button(
                                    label="📥 Download Complete Database (CSV)",
                                    data=file.read(),
                                    file_name=CSV_FILE,
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"Error reading CSV file: {e}")
    
    except Exception as e:
        st.error(f"Error fetching commits: {e}")
