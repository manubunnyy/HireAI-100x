"""
Candidate Verification Module
Searches for candidates on various platforms and enriches their profiles
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote_plus, urlparse
from github import Github
import streamlit as st

# Check if Ollama is available
def check_ollama_available():
    """Check if Ollama is running locally"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

OLLAMA_AVAILABLE = check_ollama_available()

class CandidateVerifier:
    def __init__(self, groq_api_key: str = None, use_local_llm: bool = False, ollama_model: str = "llama3.1:8b"):
        """
        Initialize the candidate verifier
        
        Args:
            groq_api_key: Groq API key for cloud-based LLM
            use_local_llm: Whether to use local Ollama model
            ollama_model: Ollama model name (default: llama3.1:8b)
        """
        self.groq_api_key = groq_api_key
        self.use_local_llm = use_local_llm and OLLAMA_AVAILABLE
        self.ollama_model = ollama_model
        
        if self.use_local_llm and not OLLAMA_AVAILABLE:
            st.warning("Ollama is not running. Please start Ollama with: `ollama serve`")
            self.use_local_llm = False
    
    def extract_emails_from_text(self, text: str) -> List[str]:
        """
        Extract email addresses from text
        """
        # Enhanced email regex pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Remove duplicates and filter valid emails
        unique_emails = list(set(emails))
        valid_emails = []
        
        for email in unique_emails:
            # Basic validation
            if '@' in email and '.' in email.split('@')[1]:
                # Filter out common non-personal emails
                if not any(prefix in email.lower() for prefix in ['noreply', 'no-reply', 'donotreply', 'info@', 'support@', 'admin@']):
                    valid_emails.append(email.lower())
        
        return valid_emails
    
    def extract_gmail_addresses(self, text: str) -> List[str]:
        """
        Extract specifically Gmail addresses from text
        """
        emails = self.extract_emails_from_text(text)
        gmail_addresses = [email for email in emails if email.endswith('@gmail.com')]
        return gmail_addresses
    
    def validate_email_domain(self, email: str) -> Dict[str, Any]:
        """
        Validate email domain and check if it's a legitimate domain
        """
        try:
            domain = email.split('@')[1]
            
            # Check if domain has MX records (indicates it can receive emails)
            import dns.resolver
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                has_mx = len(mx_records) > 0
            except:
                has_mx = False
            
            # Categorize email type
            email_type = "personal"
            if domain in ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'protonmail.com']:
                email_type = "personal"
            elif domain.endswith('.edu'):
                email_type = "educational"
            elif any(corp in domain for corp in ['.com', '.org', '.net']) and domain not in ['gmail.com', 'yahoo.com']:
                email_type = "corporate"
            
            return {
                "valid": has_mx,
                "domain": domain,
                "email_type": email_type,
                "email": email
            }
        except:
            return {
                "valid": False,
                "domain": None,
                "email_type": "unknown",
                "email": email
            }
    
    def extract_urls_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract URLs from text, categorizing them by platform
        """
        urls = {
            "github": [],
            "linkedin": [],
            "portfolio": [],
            "social": [],
            "other": []
        }
        
        # Regex pattern for URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        # Find all URLs
        found_urls = re.findall(url_pattern, text)
        
        # Also look for URLs without http/https
        domain_patterns = [
            r'(?:www\.)?github\.com/[^\s<>"{}|\\^`\[\]]*',
            r'(?:www\.)?linkedin\.com/[^\s<>"{}|\\^`\[\]]*',
            r'(?:www\.)?[a-zA-Z0-9\-]+\.(?:com|org|net|io|dev|me)/[^\s<>"{}|\\^`\[\]]*'
        ]
        
        for pattern in domain_patterns:
            found_domains = re.findall(pattern, text)
            for domain in found_domains:
                if not domain.startswith('http'):
                    found_urls.append(f'https://{domain}')
        
        # Categorize URLs
        for url in found_urls:
            url_lower = url.lower()
            if 'github.com' in url_lower:
                urls["github"].append(url)
            elif 'linkedin.com' in url_lower:
                urls["linkedin"].append(url)
            elif any(portfolio in url_lower for portfolio in ['portfolio', 'website', '.io', '.dev', '.me', 'behance', 'dribbble']):
                urls["portfolio"].append(url)
            elif any(social in url_lower for social in ['twitter.com', 'facebook.com', 'instagram.com', 'medium.com']):
                urls["social"].append(url)
            else:
                urls["other"].append(url)
        
        # Remove duplicates
        for key in urls:
            urls[key] = list(set(urls[key]))
        
        return urls
    
    def validate_github_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a GitHub URL and extract username
        """
        try:
            # Parse the URL
            parsed = urlparse(url)
            if 'github.com' not in parsed.netloc:
                return False, None
            
            # Extract username from path
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) >= 1 and path_parts[0]:
                username = path_parts[0]
                # Verify the user exists
                test_url = f"https://api.github.com/users/{username}"
                response = requests.get(test_url, headers={"Accept": "application/vnd.github.v3+json"})
                if response.status_code == 200:
                    return True, username
            
            return False, None
        except:
            return False, None
    
    def validate_linkedin_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate a LinkedIn URL (basic validation as LinkedIn requires authentication for full validation)
        """
        try:
            parsed = urlparse(url)
            if 'linkedin.com' in parsed.netloc and '/in/' in parsed.path:
                return True, url
            return False, url
        except:
            return False, url
    
    def search_github(self, name: str, email: str = None) -> Dict[str, Any]:
        """
        Search for a candidate on GitHub
        
        Returns:
            Dict with GitHub profile information
        """
        try:
            # Search by email if provided
            if email:
                # Basic GitHub search (without authentication)
                search_url = f"https://api.github.com/search/users?q={quote_plus(email)}+in:email"
                headers = {"Accept": "application/vnd.github.v3+json"}
                
                response = requests.get(search_url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("total_count", 0) > 0:
                        user = data["items"][0]
                        return self._get_github_user_details(user["login"])
            
            # Search by name
            name_parts = name.split()
            search_query = "+".join(name_parts)
            search_url = f"https://api.github.com/search/users?q={search_query}+in:name"
            headers = {"Accept": "application/vnd.github.v3+json"}
            
            response = requests.get(search_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data.get("total_count", 0) > 0:
                    # Return the most relevant result
                    user = data["items"][0]
                    return self._get_github_user_details(user["login"])
            
            return {"found": False, "platform": "GitHub"}
            
        except Exception as e:
            return {"found": False, "error": str(e), "platform": "GitHub"}
    
    def _get_github_user_details(self, username: str) -> Dict[str, Any]:
        """Get detailed information about a GitHub user"""
        try:
            user_url = f"https://api.github.com/users/{username}"
            repos_url = f"https://api.github.com/users/{username}/repos"
            headers = {"Accept": "application/vnd.github.v3+json"}
            
            # Get user info
            user_response = requests.get(user_url, headers=headers)
            if user_response.status_code != 200:
                return {"found": False, "platform": "GitHub"}
            
            user_data = user_response.json()
            
            # Get repositories
            repos_response = requests.get(repos_url, headers=headers)
            repos_data = repos_response.json() if repos_response.status_code == 200 else []
            
            # Extract relevant information
            profile = {
                "found": True,
                "platform": "GitHub",
                "username": username,
                "name": user_data.get("name", ""),
                "bio": user_data.get("bio", ""),
                "location": user_data.get("location", ""),
                "company": user_data.get("company", ""),
                "blog": user_data.get("blog", ""),
                "email": user_data.get("email", ""),
                "public_repos": user_data.get("public_repos", 0),
                "followers": user_data.get("followers", 0),
                "following": user_data.get("following", 0),
                "created_at": user_data.get("created_at", ""),
                "profile_url": user_data.get("html_url", ""),
                "avatar_url": user_data.get("avatar_url", ""),
                "repos": []
            }
            
            # Add top repositories (sorted by stars)
            if repos_data:
                sorted_repos = sorted(repos_data, key=lambda x: x.get("stargazers_count", 0), reverse=True)[:5]
                for repo in sorted_repos:
                    profile["repos"].append({
                        "name": repo.get("name", ""),
                        "description": repo.get("description", ""),
                        "language": repo.get("language", ""),
                        "stars": repo.get("stargazers_count", 0),
                        "forks": repo.get("forks_count", 0),
                        "url": repo.get("html_url", "")
                    })
            
            # Calculate programming languages used
            languages = {}
            for repo in repos_data:
                lang = repo.get("language")
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1
            profile["languages"] = languages
            
            return profile
            
        except Exception as e:
            return {"found": False, "error": str(e), "platform": "GitHub"}
    
    def search_linkedin_google(self, name: str, current_company: str = None) -> Dict[str, Any]:
        """
        Search for LinkedIn profile using Google search (since LinkedIn API requires authentication)
        
        Returns:
            Dict with LinkedIn search results
        """
        try:
            # Construct search query
            search_query = f'site:linkedin.com/in/ "{name}"'
            if current_company:
                search_query += f' "{current_company}"'
            
            # Use Google search
            search_url = f"https://www.google.com/search?q={quote_plus(search_query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(search_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find LinkedIn URLs in search results
                linkedin_urls = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if 'linkedin.com/in/' in href:
                        # Extract the actual URL from Google's redirect
                        if '/url?q=' in href:
                            actual_url = href.split('/url?q=')[1].split('&')[0]
                            linkedin_urls.append(actual_url)
                
                if linkedin_urls:
                    return {
                        "found": True,
                        "platform": "LinkedIn",
                        "possible_profiles": linkedin_urls[:3],  # Return top 3 matches
                        "search_query": search_query
                    }
            
            return {"found": False, "platform": "LinkedIn"}
            
        except Exception as e:
            return {"found": False, "error": str(e), "platform": "LinkedIn"}
    
    def analyze_online_presence(self, candidate_info: Dict[str, Any], online_profiles: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to analyze and summarize the candidate's online presence
        """
        # Prepare the analysis prompt
        prompt = f"""
        Analyze this candidate's online presence and provide insights:
        
        Candidate Information from Resume:
        Name: {candidate_info.get('name', 'Unknown')}
        Email: {candidate_info.get('email', 'Not provided')}
        Skills: {', '.join(candidate_info.get('skills', []))}
        
        Online Profiles Found:
        {json.dumps(online_profiles, indent=2)}
        
        Please provide:
        1. Verification confidence (0-100%) that these profiles belong to the same person
        2. Additional skills or technologies found online but not in resume
        3. Professional activity level (based on GitHub contributions, etc.)
        4. Any red flags or concerns
        5. Overall assessment of online presence
        
        Format your response as JSON with keys: 
        verification_confidence, additional_skills, activity_level, red_flags, assessment
        """
        
        if self.use_local_llm:
            # Use Ollama model
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    analysis_text = response.json()["response"]
                else:
                    return {"error": f"Ollama error: {response.status_code}"}
                    
            except Exception as e:
                return {"error": f"Ollama error: {str(e)}"}
        else:
            # Use Groq API
            try:
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "llama3-70b-8192",
                    "temperature": 0.1,
                    "max_tokens": 512
                }
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    analysis_text = response.json()["choices"][0]["message"]["content"]
                else:
                    return {"error": f"API error: {response.status_code}"}
                    
            except Exception as e:
                return {"error": f"API error: {str(e)}"}
        
        # Parse the AI response
        try:
            # Extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', analysis_text)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback: create structured response from text
                analysis = {
                    "verification_confidence": 50,
                    "additional_skills": [],
                    "activity_level": "Unknown",
                    "red_flags": [],
                    "assessment": analysis_text
                }
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Failed to parse AI response: {str(e)}",
                "raw_response": analysis_text
            }
    
    def verify_candidate(self, candidate_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to verify a candidate across multiple platforms
        
        Args:
            candidate_info: Dictionary containing candidate's resume information
            
        Returns:
            Dictionary with verification results
        """
        name = candidate_info.get("name", "")
        email = candidate_info.get("email", "")
        
        if not name:
            return {"error": "No candidate name provided"}
        
        results = {
            "candidate_name": name,
            "verification_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profiles": {}
        }
        
        # Search GitHub
        with st.spinner(f"Searching GitHub for {name}..."):
            github_result = self.search_github(name, email)
            results["profiles"]["github"] = github_result
        
        # Search LinkedIn via Google
        with st.spinner(f"Searching LinkedIn for {name}..."):
            company = None
            if candidate_info.get("experience"):
                # Try to extract current company from experience
                exp_list = candidate_info["experience"]
                if exp_list and isinstance(exp_list, list) and len(exp_list) > 0:
                    # Assume first experience is most recent
                    company = exp_list[0].split(" at ")[-1] if " at " in exp_list[0] else None
            
            linkedin_result = self.search_linkedin_google(name, company)
            results["profiles"]["linkedin"] = linkedin_result
        
        # AI Analysis of online presence
        with st.spinner("Analyzing online presence with AI..."):
            ai_analysis = self.analyze_online_presence(candidate_info, results["profiles"])
            results["ai_analysis"] = ai_analysis
        
        # Summary
        profiles_found = sum(1 for p in results["profiles"].values() if p.get("found", False))
        results["summary"] = {
            "profiles_found": profiles_found,
            "platforms_checked": len(results["profiles"]),
            "verification_status": "Verified" if profiles_found > 0 else "Not Found"
        }
        
        return results
    
    def verify_candidate_with_urls(self, candidate_info: Dict[str, Any], resume_text: str, embedded_links: List[str] = None) -> Dict[str, Any]:
        """
        Enhanced verification that extracts and validates URLs from resume
        
        Args:
            candidate_info: Dictionary containing candidate's resume information
            resume_text: Raw text from resume for URL extraction
            embedded_links: List of embedded links extracted from PDF
            
        Returns:
            Dictionary with comprehensive verification results
        """
        name = candidate_info.get("name", "")
        email = candidate_info.get("email", "")
        
        if not name:
            return {"error": "No candidate name provided"}
        
        results = {
            "candidate_name": name,
            "verification_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profiles": {},
            "extracted_urls": {},
            "embedded_links": embedded_links or [],
            "validation_status": {},
            "scraped_content": {}
        }
        
        # Extract URLs from resume text
        st.info("🔍 Extracting URLs from resume text...")
        extracted_urls = self.extract_urls_from_text(resume_text)
        results["extracted_urls"] = extracted_urls
        
        # Combine embedded links with extracted URLs
        all_urls = []
        for url_list in extracted_urls.values():
            all_urls.extend(url_list)
        
        # Add embedded links if provided
        if embedded_links:
            st.info(f"📎 Found {len(embedded_links)} embedded links in PDF")
            all_urls.extend(embedded_links)
            
            # Categorize embedded links
            for link in embedded_links:
                link_lower = link.lower()
                if 'github.com' in link_lower and link not in extracted_urls["github"]:
                    extracted_urls["github"].append(link)
                elif 'linkedin.com' in link_lower and link not in extracted_urls["linkedin"]:
                    extracted_urls["linkedin"].append(link)
                elif any(portfolio in link_lower for portfolio in ['portfolio', 'website', '.io', '.dev', '.me', 'behance', 'dribbble']):
                    if link not in extracted_urls["portfolio"]:
                        extracted_urls["portfolio"].append(link)
                elif any(social in link_lower for social in ['twitter.com', 'facebook.com', 'instagram.com', 'medium.com']):
                    if link not in extracted_urls["social"]:
                        extracted_urls["social"].append(link)
                else:
                    if link not in extracted_urls["other"]:
                        extracted_urls["other"].append(link)
        
        # Remove duplicates from all URLs
        all_urls = list(set(all_urls))
        
        # Scrape all found URLs
        if all_urls:
            st.info(f"🌐 Scraping content from {len(all_urls)} URLs...")
            progress_bar = st.progress(0)
            
            for idx, url in enumerate(all_urls):
                progress_bar.progress((idx + 1) / len(all_urls))
                
                # Skip if already processed
                if url in results["scraped_content"]:
                    continue
                
                with st.spinner(f"Scraping: {url[:50]}..."):
                    scraped_data = self.scrape_url_content(url)
                    results["scraped_content"][url] = scraped_data
                    
                    # Log success/failure
                    if scraped_data.get("success"):
                        st.success(f"✅ Successfully scraped: {scraped_data.get('platform', 'Unknown')} - {scraped_data.get('title', url)[:50]}")
                    else:
                        st.warning(f"❌ Failed to scrape: {url[:50]} - {scraped_data.get('error', 'Unknown error')}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            progress_bar.empty()
        
        # Process GitHub URLs specifically for detailed profile info
        github_profile = None
        if extracted_urls["github"]:
            st.info(f"Found {len(extracted_urls['github'])} GitHub URL(s)")
            for url in extracted_urls["github"]:
                is_valid, username = self.validate_github_url(url)
                if is_valid and username:
                    st.success(f"✅ Valid GitHub profile found: {username}")
                    github_profile = self._get_github_user_details(username)
                    results["validation_status"]["github"] = "verified"
                    break
                else:
                    st.warning(f"❌ Invalid GitHub URL: {url}")
        
        # If no valid GitHub URL found, search by name/email
        if not github_profile:
            with st.spinner(f"Searching GitHub for {name}..."):
                github_profile = self.search_github(name, email)
                results["validation_status"]["github"] = "searched"
        
        results["profiles"]["github"] = github_profile
        
        # Process LinkedIn URLs
        linkedin_profile = {"found": False, "platform": "LinkedIn"}
        if extracted_urls["linkedin"]:
            st.info(f"Found {len(extracted_urls['linkedin'])} LinkedIn URL(s)")
            valid_urls = []
            scraped_linkedin_data = []
            
            for url in extracted_urls["linkedin"]:
                is_valid, clean_url = self.validate_linkedin_url(url)
                if is_valid:
                    valid_urls.append(clean_url)
                    # Get scraped data for this LinkedIn URL
                    if url in results["scraped_content"] and results["scraped_content"][url].get("success"):
                        scraped_linkedin_data.append(results["scraped_content"][url])
            
            if valid_urls:
                linkedin_profile = {
                    "found": True,
                    "platform": "LinkedIn",
                    "verified_urls": valid_urls,
                    "scraped_data": scraped_linkedin_data,
                    "validation_status": "verified"
                }
                results["validation_status"]["linkedin"] = "verified"
                st.success(f"✅ Valid LinkedIn profile(s) found")
        else:
            # Search for LinkedIn via Google if no URLs found
            with st.spinner(f"Searching LinkedIn for {name}..."):
                company = None
                if candidate_info.get("experience"):
                    exp_list = candidate_info["experience"]
                    if exp_list and isinstance(exp_list, list) and len(exp_list) > 0:
                        company = exp_list[0].split(" at ")[-1] if " at " in exp_list[0] else None
                
                linkedin_profile = self.search_linkedin_google(name, company)
                results["validation_status"]["linkedin"] = "searched"
        
        results["profiles"]["linkedin"] = linkedin_profile
        
        # Process portfolio and other professional sites
        portfolio_data = []
        if extracted_urls["portfolio"]:
            st.info(f"Found {len(extracted_urls['portfolio'])} portfolio/professional site(s)")
            for url in extracted_urls["portfolio"]:
                if url in results["scraped_content"] and results["scraped_content"][url].get("success"):
                    portfolio_data.append(results["scraped_content"][url])
        
        if portfolio_data:
            results["profiles"]["portfolio"] = {
                "found": True,
                "platform": "Portfolio Sites",
                "sites": portfolio_data
            }
        
        # Process social media links
        social_data = []
        if extracted_urls["social"]:
            st.info(f"Found {len(extracted_urls['social'])} social media link(s)")
            for url in extracted_urls["social"]:
                if url in results["scraped_content"] and results["scraped_content"][url].get("success"):
                    social_data.append(results["scraped_content"][url])
        
        if social_data:
            results["profiles"]["social_media"] = {
                "found": True,
                "platform": "Social Media",
                "profiles": social_data
            }
        
        # AI Analysis with enhanced context including scraped content
        with st.spinner("Analyzing online presence and authenticity with scraped data..."):
            ai_analysis = self.analyze_enhanced_presence_with_scraping(
                candidate_info, 
                results["profiles"], 
                extracted_urls,
                results["scraped_content"]
            )
            results["ai_analysis"] = ai_analysis
        
        # Authenticity score with scraped content consideration
        authenticity_score = self._calculate_authenticity_score_enhanced(results)
        results["authenticity_score"] = authenticity_score
        
        # Summary
        profiles_found = sum(1 for p in results["profiles"].values() if p.get("found", False))
        urls_scraped = sum(1 for s in results["scraped_content"].values() if s.get("success", False))
        
        results["summary"] = {
            "profiles_found": profiles_found,
            "platforms_checked": len(results["profiles"]),
            "urls_scraped": urls_scraped,
            "total_urls_found": len(all_urls),
            "verification_status": "Verified" if authenticity_score >= 70 else "Needs Review",
            "authenticity_score": authenticity_score
        }
        
        return results
    
    def _calculate_authenticity_score(self, verification_results: Dict[str, Any]) -> int:
        """
        Calculate an authenticity score based on verification results
        """
        score = 0
        
        # Points for verified URLs (more trustworthy)
        if verification_results.get("validation_status", {}).get("github") == "verified":
            score += 40
        elif verification_results["profiles"].get("github", {}).get("found"):
            score += 20
        
        if verification_results.get("validation_status", {}).get("linkedin") == "verified":
            score += 30
        elif verification_results["profiles"].get("linkedin", {}).get("found"):
            score += 15
        
        # AI confidence also contributes
        ai_confidence = verification_results.get("ai_analysis", {}).get("verification_confidence", 0)
        score += int(ai_confidence * 0.3)  # Up to 30 points from AI confidence
        
        return min(100, score)
    
    def analyze_enhanced_presence(self, candidate_info: Dict[str, Any], online_profiles: Dict[str, Any], extracted_urls: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Enhanced AI analysis including URL validation results
        """
        prompt = f"""
        Perform a comprehensive authenticity and background check for this candidate:
        
        Candidate Information from Resume:
        Name: {candidate_info.get('name', 'Unknown')}
        Email: {candidate_info.get('email', 'Not provided')}
        Skills: {', '.join(candidate_info.get('skills', []))}
        Projects: {json.dumps(candidate_info.get('projects', []))}
        
        URLs Found in Resume:
        {json.dumps(extracted_urls, indent=2)}
        
        Online Profiles Verified:
        {json.dumps(online_profiles, indent=2)}
        
        Analyze and provide:
        1. Verification confidence (0-100%) that this is a genuine candidate
        2. Consistency check between resume claims and online presence
        3. Red flags or concerns about authenticity
        4. Additional insights from online profiles
        5. Overall trust assessment
        
        Consider:
        - Do the URLs in resume actually lead to their profiles?
        - Is the online activity consistent with claimed experience?
        - Any discrepancies between resume and online profiles?
        
        Format response as JSON:
        {{
            "verification_confidence": <0-100>,
            "consistency_score": <0-100>,
            "red_flags": ["flag1", "flag2"],
            "positive_signals": ["signal1", "signal2"],
            "additional_insights": ["insight1", "insight2"],
            "trust_assessment": "assessment text"
        }}
        """
        
        # Use the existing analyze_online_presence logic but with enhanced prompt
        return self._call_llm_for_analysis(prompt)
    
    def _call_llm_for_analysis(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM (Ollama or Groq) for analysis
        """
        if self.use_local_llm:
            # Use Ollama model
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    analysis_text = response.json()["response"]
                else:
                    return {"error": f"Ollama error: {response.status_code}"}
                    
            except Exception as e:
                return {"error": f"Ollama error: {str(e)}"}
        else:
            # Use Groq API
            try:
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "llama3-70b-8192",
                    "temperature": 0.1,
                    "max_tokens": 512
                }
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    analysis_text = response.json()["choices"][0]["message"]["content"]
                else:
                    return {"error": f"API error: {response.status_code}"}
                    
            except Exception as e:
                return {"error": f"API error: {str(e)}"}
        
        # Parse the response
        try:
            json_match = re.search(r'\{[\s\S]*\}', analysis_text)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse AI response", "raw": analysis_text}
        except Exception as e:
            return {"error": f"Parse error: {str(e)}", "raw": analysis_text}
    
    def generate_prescreening_questions(self, candidate_info: Dict[str, Any], job_requirements: Dict[str, Any], verification_results: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Generate intelligent pre-screening questions based on resume, job requirements, and verification results
        """
        prompt = f"""
        Generate 5-7 intelligent pre-screening interview questions for this candidate.
        
        Candidate Profile:
        Name: {candidate_info.get('name', 'Unknown')}
        Skills: {', '.join(candidate_info.get('skills', []))}
        Projects: {json.dumps(candidate_info.get('projects', []))}
        Experience: {json.dumps(candidate_info.get('experience', []))}
        
        Job Requirements:
        {json.dumps(job_requirements, indent=2)}
        
        {"Online Verification Results:" + json.dumps(verification_results.get('ai_analysis', {}), indent=2) if verification_results else ""}
        
        Generate questions that:
        1. Verify claims made in the resume
        2. Assess technical skills relevant to the job
        3. Explore project experience in detail
        4. Check cultural fit and soft skills
        5. Address any gaps or concerns
        
        Format as JSON array:
        [
            {{
                "question": "question text",
                "purpose": "what this question assesses",
                "expected_insights": "what a good answer would reveal"
            }}
        ]
        
        Make questions specific and probing, not generic.
        """
        
        result = self._call_llm_for_analysis(prompt)
        
        if "error" in result:
            # Fallback questions
            return [
                {
                    "question": f"Can you describe your experience with {candidate_info.get('skills', ['relevant technologies'])[0] if candidate_info.get('skills') else 'the required technologies'}?",
                    "purpose": "Assess technical depth",
                    "expected_insights": "Technical proficiency and hands-on experience"
                },
                {
                    "question": "Tell me about your most challenging project. What was your role and how did you overcome the challenges?",
                    "purpose": "Evaluate problem-solving skills",
                    "expected_insights": "Problem-solving approach and technical leadership"
                }
            ]
        
        # Parse the response
        try:
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "questions" in result:
                return result["questions"]
            else:
                # Try to extract array from text
                text = str(result)
                array_match = re.search(r'\[[\s\S]*\]', text)
                if array_match:
                    return json.loads(array_match.group())
                return []
        except:
            return []

    def scrape_url_content(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from any URL and extract relevant information
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Determine the type of website
            domain = urlparse(url).netloc.lower()
            
            result = {
                "url": url,
                "title": soup.title.string if soup.title else "",
                "success": True
            }
            
            # Special handling for known platforms
            if 'linkedin.com' in domain:
                result.update(self._scrape_linkedin_profile(soup, url))
            elif 'github.com' in domain:
                result.update(self._scrape_github_profile(soup, url))
            elif 'behance.net' in domain:
                result.update(self._scrape_behance_profile(soup))
            elif 'dribbble.com' in domain:
                result.update(self._scrape_dribbble_profile(soup))
            elif 'medium.com' in domain:
                result.update(self._scrape_medium_profile(soup))
            elif any(portfolio in domain for portfolio in ['wixsite.com', 'squarespace.com', 'wordpress.com', '.github.io']):
                result.update(self._scrape_portfolio_site(soup))
            else:
                # Generic scraping for unknown sites
                result.update(self._scrape_generic_site(soup))
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {
                "url": url,
                "success": False,
                "error": f"Failed to fetch URL: {str(e)}"
            }
        except Exception as e:
            return {
                "url": url,
                "success": False,
                "error": f"Error scraping URL: {str(e)}"
            }
    
    def _scrape_linkedin_profile(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract information from LinkedIn profile page (limited due to auth requirements)"""
        data = {"platform": "LinkedIn"}
        
        # Extract what's available without login
        # LinkedIn heavily restricts public access, so we get limited info
        
        # Try to extract name from page title
        if soup.title:
            title_text = soup.title.string
            # LinkedIn titles usually follow pattern "Name - Title | LinkedIn"
            if " - " in title_text and " | LinkedIn" in title_text:
                parts = title_text.split(" - ")
                if parts:
                    data["name"] = parts[0].strip()
                    if len(parts) > 1:
                        data["headline"] = parts[1].split(" | LinkedIn")[0].strip()
        
        # Try to extract any visible text that might contain info
        meta_description = soup.find("meta", {"name": "description"})
        if meta_description:
            data["description"] = meta_description.get("content", "")
        
        # Note about limitations
        data["note"] = "LinkedIn requires authentication for full profile access"
        data["profile_url"] = url
        
        return data
    
    def _scrape_github_profile(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract information from GitHub profile page"""
        data = {"platform": "GitHub"}
        
        # Extract username from URL
        path_parts = urlparse(url).path.strip('/').split('/')
        if path_parts:
            username = path_parts[0]
            data["username"] = username
            
            # If it's a repo page
            if len(path_parts) >= 2:
                data["repo_name"] = path_parts[1]
                
                # Extract repo description
                about_section = soup.find("p", {"class": "f4"})
                if about_section:
                    data["repo_description"] = about_section.text.strip()
                
                # Extract languages
                languages = soup.find_all("span", {"class": "color-fg-default text-bold mr-1"})
                if languages:
                    data["languages"] = [lang.text.strip() for lang in languages]
                
                # Extract stars
                star_element = soup.find("a", {"href": f"/{username}/{path_parts[1]}/stargazers"})
                if star_element:
                    star_text = star_element.find("span")
                    if star_text:
                        data["stars"] = star_text.text.strip()
            else:
                # Profile page
                # Extract name
                name_element = soup.find("span", {"class": "p-name"})
                if name_element:
                    data["name"] = name_element.text.strip()
                
                # Extract bio
                bio_element = soup.find("div", {"class": "p-note"})
                if bio_element:
                    data["bio"] = bio_element.text.strip()
                
                # Extract location
                location_element = soup.find("span", {"class": "p-label"})
                if location_element:
                    data["location"] = location_element.text.strip()
                
                # Extract organizations
                org_elements = soup.find_all("a", {"class": "avatar-group-item"})
                if org_elements:
                    data["organizations"] = [org.get("aria-label", "") for org in org_elements]
        
        return data
    
    def _scrape_behance_profile(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract information from Behance profile"""
        data = {"platform": "Behance"}
        
        # Extract profile name
        name_element = soup.find("h1", {"class": "ProfileCard-userFullName"})
        if name_element:
            data["name"] = name_element.text.strip()
        
        # Extract profile info
        info_element = soup.find("p", {"class": "ProfileCard-bio"})
        if info_element:
            data["bio"] = info_element.text.strip()
        
        # Extract location
        location_element = soup.find("span", {"class": "ProfileCard-location"})
        if location_element:
            data["location"] = location_element.text.strip()
        
        # Extract stats
        stats = soup.find_all("div", {"class": "ProfileCard-stat"})
        if stats:
            data["stats"] = {}
            for stat in stats:
                label = stat.find("span", {"class": "ProfileCard-statLabel"})
                value = stat.find("span", {"class": "ProfileCard-statValue"})
                if label and value:
                    data["stats"][label.text.strip()] = value.text.strip()
        
        return data
    
    def _scrape_dribbble_profile(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract information from Dribbble profile"""
        data = {"platform": "Dribbble"}
        
        # Extract name
        name_element = soup.find("h1", {"class": "profile-name"})
        if name_element:
            data["name"] = name_element.text.strip()
        
        # Extract bio
        bio_element = soup.find("div", {"class": "profile-bio"})
        if bio_element:
            data["bio"] = bio_element.text.strip()
        
        # Extract location
        location_element = soup.find("span", {"class": "location"})
        if location_element:
            data["location"] = location_element.text.strip()
        
        return data
    
    def _scrape_medium_profile(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract information from Medium profile"""
        data = {"platform": "Medium"}
        
        # Extract name
        name_element = soup.find("h2", {"class": re.compile("pw-author-name")})
        if name_element:
            data["name"] = name_element.text.strip()
        
        # Extract bio
        bio_element = soup.find("p", {"class": re.compile("pw-bio")})
        if bio_element:
            data["bio"] = bio_element.text.strip()
        
        # Extract follower count
        followers = soup.find("span", string=re.compile(r"\d+.*Followers"))
        if followers:
            data["followers"] = followers.text.strip()
        
        return data
    
    def _scrape_portfolio_site(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract information from portfolio websites"""
        data = {"platform": "Portfolio"}
        
        # Extract title
        if soup.title:
            data["site_title"] = soup.title.string
        
        # Look for common patterns in portfolio sites
        
        # Try to find name in various common locations
        for selector in ["h1", "h2", ".name", ".author", "#name", ".hero-title"]:
            name_element = soup.find(selector)
            if name_element and len(name_element.text.strip()) < 50:  # Reasonable name length
                data["possible_name"] = name_element.text.strip()
                break
        
        # Look for about section
        for selector in [".about", "#about", ".bio", ".description", "section#about"]:
            about_element = soup.find(selector)
            if about_element:
                data["about"] = about_element.text.strip()[:500]  # Limit length
                break
        
        # Look for skills
        skills_section = soup.find_all(["div", "section", "ul"], {"class": re.compile(r"skill|expertise|competenc")})
        if skills_section:
            skills = []
            for section in skills_section[:2]:  # Limit to first 2 sections
                skill_items = section.find_all(["li", "span", "div"])
                skills.extend([item.text.strip() for item in skill_items[:10]])  # Limit items
            if skills:
                data["skills"] = list(set(skills))[:20]  # Remove duplicates, limit to 20
        
        # Look for contact information
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, soup.text)
        if emails:
            data["contact_emails"] = list(set(emails))[:3]  # Limit to 3 emails
        
        return data
    
    def _scrape_generic_site(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Generic scraping for unknown websites"""
        data = {"platform": "Generic Website"}
        
        # Extract meta information
        meta_data = {}
        for meta in soup.find_all("meta"):
            if meta.get("name"):
                meta_data[meta.get("name")] = meta.get("content", "")
            elif meta.get("property"):
                meta_data[meta.get("property")] = meta.get("content", "")
        
        if meta_data:
            data["meta_info"] = meta_data
        
        # Extract main headings
        headings = []
        for h in soup.find_all(["h1", "h2"]):
            text = h.text.strip()
            if text and len(text) < 100:  # Reasonable heading length
                headings.append(text)
        
        if headings:
            data["main_headings"] = headings[:5]  # Limit to 5 headings
        
        # Look for social media links
        social_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            for social in ["linkedin.com", "github.com", "twitter.com", "facebook.com", "instagram.com"]:
                if social in href:
                    social_links.append({"platform": social.split(".")[0], "url": a["href"]})
                    break
        
        if social_links:
            data["social_links"] = social_links[:10]  # Limit to 10 links
        
        # Extract any structured data (JSON-LD)
        json_ld = soup.find("script", {"type": "application/ld+json"})
        if json_ld:
            try:
                structured_data = json.loads(json_ld.string)
                data["structured_data"] = structured_data
            except:
                pass
        
        return data

    def analyze_enhanced_presence_with_scraping(self, candidate_info: Dict[str, Any], online_profiles: Dict[str, Any], 
                                               extracted_urls: Dict[str, List[str]], scraped_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced AI analysis including scraped content from all URLs
        """
        # Prepare scraped content summary
        scraped_summary = []
        for url, content in scraped_content.items():
            if content.get("success"):
                summary_item = {
                    "url": url,
                    "platform": content.get("platform", "Unknown"),
                    "title": content.get("title", "")
                }
                
                # Add relevant extracted information
                if content.get("name"):
                    summary_item["name"] = content.get("name")
                if content.get("bio"):
                    summary_item["bio"] = content.get("bio")[:200]  # Limit length
                if content.get("skills"):
                    summary_item["skills"] = content.get("skills")[:10]  # Limit to 10 skills
                if content.get("location"):
                    summary_item["location"] = content.get("location")
                
                scraped_summary.append(summary_item)
        
        prompt = f"""
        Perform a comprehensive authenticity and background check for this candidate using all available data:
        
        Candidate Information from Resume:
        Name: {candidate_info.get('name', 'Unknown')}
        Email: {candidate_info.get('email', 'Not provided')}
        Skills: {', '.join(candidate_info.get('skills', [])[:20])}
        Projects: {json.dumps(candidate_info.get('projects', [])[:5])}
        
        URLs Found in Resume:
        {json.dumps(extracted_urls, indent=2)}
        
        Online Profiles Verified:
        {json.dumps(online_profiles, indent=2)}
        
        Content Scraped from URLs:
        {json.dumps(scraped_summary, indent=2)}
        
        Analyze and provide:
        1. Verification confidence (0-100%) that this is a genuine candidate
        2. Consistency check between resume claims and scraped online content
        3. Red flags or concerns about authenticity
        4. Additional skills or information found online but not in resume
        5. Overall trust assessment based on all scraped data
        6. Professional activity level based on online presence
        
        Consider:
        - Do the URLs in resume actually lead to profiles about the same person?
        - Is the information consistent across different platforms?
        - Are there discrepancies in skills, experience, or education?
        - Is the online activity recent and relevant?
        - Do portfolio sites showcase work that matches claimed skills?
        
        Format response as JSON:
        {{
            "verification_confidence": <0-100>,
            "consistency_score": <0-100>,
            "red_flags": ["flag1", "flag2"],
            "positive_signals": ["signal1", "signal2"],
            "additional_skills_found": ["skill1", "skill2"],
            "additional_insights": ["insight1", "insight2"],
            "professional_activity": "Active/Moderate/Low",
            "trust_assessment": "detailed assessment text"
        }}
        """
        
        return self._call_llm_for_analysis(prompt)
    
    def _calculate_authenticity_score_enhanced(self, verification_results: Dict[str, Any]) -> int:
        """
        Calculate an enhanced authenticity score based on verification results and scraped content
        """
        score = 0
        
        # Points for verified URLs (more trustworthy)
        if verification_results.get("validation_status", {}).get("github") == "verified":
            score += 25
        elif verification_results["profiles"].get("github", {}).get("found"):
            score += 15
        
        if verification_results.get("validation_status", {}).get("linkedin") == "verified":
            score += 20
        elif verification_results["profiles"].get("linkedin", {}).get("found"):
            score += 10
        
        # Points for successfully scraped URLs
        scraped_content = verification_results.get("scraped_content", {})
        successful_scrapes = sum(1 for s in scraped_content.values() if s.get("success", False))
        total_urls = len(scraped_content)
        
        if total_urls > 0:
            scrape_success_rate = successful_scrapes / total_urls
            score += int(scrape_success_rate * 20)  # Up to 20 points for successful scraping
        
        # Points for portfolio sites
        if verification_results["profiles"].get("portfolio", {}).get("found"):
            score += 15
        
        # Points for consistency (from AI analysis)
        ai_analysis = verification_results.get("ai_analysis", {})
        
        # AI confidence contributes
        ai_confidence = ai_analysis.get("verification_confidence", 0)
        score += int(ai_confidence * 0.2)  # Up to 20 points from AI confidence
        
        # Consistency score contributes
        consistency = ai_analysis.get("consistency_score", 0)
        score += int(consistency * 0.1)  # Up to 10 points from consistency
        
        # Deduct points for red flags
        red_flags = ai_analysis.get("red_flags", [])
        if red_flags:
            score -= len(red_flags) * 5  # -5 points per red flag
        
        # Ensure score is within 0-100 range
        return max(0, min(100, score))


def format_verification_report(verification_results: Dict[str, Any]) -> str:
    """
    Format verification results into a readable report
    """
    report = f"""
## Online Presence Verification Report

**Candidate:** {verification_results.get('candidate_name', 'Unknown')}  
**Verification Date:** {verification_results.get('verification_timestamp', 'N/A')}  
**Status:** {verification_results['summary']['verification_status']}  
**Authenticity Score:** {verification_results.get('authenticity_score', 0)}/100

### Summary
- **Profiles Found:** {verification_results['summary']['profiles_found']}/{verification_results['summary']['platforms_checked']}
- **URLs Scraped:** {verification_results['summary'].get('urls_scraped', 0)}/{verification_results['summary'].get('total_urls_found', 0)}

"""
    
    # Show embedded links if available
    if verification_results.get('embedded_links'):
        report += f"### Embedded Links in PDF ({len(verification_results['embedded_links'])})\n"
        for link in verification_results['embedded_links'][:10]:  # Limit to 10
            report += f"- {link}\n"
        if len(verification_results['embedded_links']) > 10:
            report += f"- ... and {len(verification_results['embedded_links']) - 10} more\n"
        report += "\n"
    
    # Show extracted URLs if available
    if verification_results.get('extracted_urls'):
        report += "### URLs Found in Resume Text\n"
        urls = verification_results['extracted_urls']
        if urls.get('github'):
            report += f"**GitHub:** {', '.join(urls['github'])}\n"
        if urls.get('linkedin'):
            report += f"**LinkedIn:** {', '.join(urls['linkedin'])}\n"
        if urls.get('portfolio'):
            report += f"**Portfolio:** {', '.join(urls['portfolio'][:3])}{'...' if len(urls['portfolio']) > 3 else ''}\n"
        if urls.get('social'):
            report += f"**Social Media:** {', '.join(urls['social'][:3])}{'...' if len(urls['social']) > 3 else ''}\n"
        if urls.get('other'):
            report += f"**Other:** {', '.join(urls['other'][:3])}{'...' if len(urls['other']) > 3 else ''}\n"
        report += "\n"
    
    # Scraped Content Summary
    if verification_results.get('scraped_content'):
        successful_scrapes = [url for url, content in verification_results['scraped_content'].items() if content.get('success')]
        failed_scrapes = [url for url, content in verification_results['scraped_content'].items() if not content.get('success')]
        
        if successful_scrapes:
            report += f"### Successfully Scraped URLs ({len(successful_scrapes)})\n"
            for url in successful_scrapes[:10]:
                content = verification_results['scraped_content'][url]
                platform = content.get('platform', 'Unknown')
                title = content.get('title', url)[:50]
                report += f"- **{platform}**: {title}{'...' if len(title) >= 50 else ''}\n"
                
                # Add key information if available
                if content.get('name'):
                    report += f"  - Name: {content['name']}\n"
                if content.get('bio'):
                    report += f"  - Bio: {content['bio'][:100]}...\n"
                if content.get('skills'):
                    report += f"  - Skills: {', '.join(content['skills'][:5])}{'...' if len(content['skills']) > 5 else ''}\n"
            
            if len(successful_scrapes) > 10:
                report += f"\n... and {len(successful_scrapes) - 10} more successfully scraped URLs\n"
            report += "\n"
        
        if failed_scrapes:
            report += f"### Failed to Scrape ({len(failed_scrapes)})\n"
            for url in failed_scrapes[:5]:
                error = verification_results['scraped_content'][url].get('error', 'Unknown error')
                report += f"- {url[:50]}{'...' if len(url) > 50 else ''}: {error}\n"
            if len(failed_scrapes) > 5:
                report += f"... and {len(failed_scrapes) - 5} more failed URLs\n"
            report += "\n"
    
    # GitHub Results
    github = verification_results['profiles'].get('github', {})
    if github.get('found'):
        validation = verification_results.get('validation_status', {}).get('github', 'searched')
        status_icon = "✅" if validation == "verified" else "🔍"
        report += f"""
### GitHub Profile {status_icon} ({validation})
- **Username:** [{github.get('username')}]({github.get('profile_url')})
- **Name:** {github.get('name', 'N/A')}
- **Bio:** {github.get('bio', 'N/A')}
- **Location:** {github.get('location', 'N/A')}
- **Public Repos:** {github.get('public_repos', 0)}
- **Followers:** {github.get('followers', 0)}

**Top Repositories:**
"""
        for repo in github.get('repos', [])[:3]:
            report += f"- [{repo['name']}]({repo['url']}) - {repo.get('language', 'N/A')} (⭐ {repo['stars']})\n"
        
        if github.get('languages'):
            report += "\n**Programming Languages:**\n"
            for lang, count in sorted(github['languages'].items(), key=lambda x: x[1], reverse=True)[:5]:
                report += f"- {lang}: {count} repos\n"
    else:
        report += "\n### GitHub Profile ✗\nNo GitHub profile found.\n"
    
    # LinkedIn Results
    linkedin = verification_results['profiles'].get('linkedin', {})
    if linkedin.get('found'):
        validation = verification_results.get('validation_status', {}).get('linkedin', 'searched')
        status_icon = "✅" if validation == "verified" else "🔍"
        report += f"\n### LinkedIn Profile {status_icon} ({validation})\n"
        
        if linkedin.get('verified_urls'):
            report += "**Verified LinkedIn URLs:**\n"
            for url in linkedin['verified_urls']:
                report += f"- {url}\n"
        elif linkedin.get('possible_profiles'):
            report += "**Possible LinkedIn Profiles Found:**\n"
            for url in linkedin['possible_profiles']:
                report += f"- {url}\n"
        
        # Include scraped LinkedIn data if available
        if linkedin.get('scraped_data'):
            report += "\n**Scraped LinkedIn Information:**\n"
            for data in linkedin['scraped_data']:
                if data.get('name'):
                    report += f"- Name: {data['name']}\n"
                if data.get('headline'):
                    report += f"- Headline: {data['headline']}\n"
                if data.get('description'):
                    report += f"- Description: {data['description'][:200]}...\n"
    else:
        report += "\n### LinkedIn Profile ✗\nNo LinkedIn profile found.\n"
    
    # Portfolio Sites
    portfolio = verification_results['profiles'].get('portfolio', {})
    if portfolio.get('found'):
        report += "\n### Portfolio/Professional Sites ✅\n"
        for site in portfolio.get('sites', [])[:5]:
            report += f"\n**{site.get('platform', 'Portfolio Site')}**: {site.get('url', '')[:50]}{'...' if len(site.get('url', '')) > 50 else ''}\n"
            if site.get('site_title'):
                report += f"- Title: {site['site_title']}\n"
            if site.get('possible_name'):
                report += f"- Name: {site['possible_name']}\n"
            if site.get('about'):
                report += f"- About: {site['about'][:150]}...\n"
            if site.get('skills'):
                report += f"- Skills: {', '.join(site['skills'][:10])}\n"
    
    # Enhanced AI Analysis
    ai_analysis = verification_results.get('ai_analysis', {})
    if ai_analysis and 'error' not in ai_analysis:
        report += f"""
### AI Authenticity Analysis

**Verification Confidence:** {ai_analysis.get('verification_confidence', 'N/A')}%  
**Consistency Score:** {ai_analysis.get('consistency_score', 'N/A')}%  
**Professional Activity:** {ai_analysis.get('professional_activity', 'N/A')}

**Positive Signals:**
"""
        for signal in ai_analysis.get('positive_signals', []):
            report += f"- ✅ {signal}\n"
        
        if ai_analysis.get('red_flags'):
            report += "\n**Red Flags:**\n"
            for flag in ai_analysis['red_flags']:
                report += f"- ⚠️ {flag}\n"
        
        if ai_analysis.get('additional_skills_found'):
            report += "\n**Additional Skills Found Online:**\n"
            for skill in ai_analysis['additional_skills_found']:
                report += f"- 💡 {skill}\n"
        
        if ai_analysis.get('additional_insights'):
            report += "\n**Additional Insights:**\n"
            for insight in ai_analysis['additional_insights']:
                report += f"- 💡 {insight}\n"
        
        report += f"\n**Trust Assessment:**\n{ai_analysis.get('trust_assessment', 'No assessment available.')}\n"
    
    return report 