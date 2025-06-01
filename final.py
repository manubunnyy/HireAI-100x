import streamlit as st
import pandas as pd
import os
import zipfile
import re
import fitz  # PyMuPDF
import io
from io import BytesIO
import base64
import tempfile
import json
import time
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
# from dotenv import load_dotenv  # Commented out - not using .env file
import hashlib
import pickle
import random
import threading
import queue
from datetime import datetime, timedelta

# Import the candidate verification module
from candidate_verifier import CandidateVerifier, format_verification_report

# Import the email handler module
from email_handler import EmailHandler

# Create a minimal verifier that suppresses verbose output
class SilentCandidateVerifier(CandidateVerifier):
    """A CandidateVerifier that suppresses verbose Streamlit output"""
    
    def verify_candidate_with_urls(self, candidate_info, resume_text, embedded_links=None):
        """Override to suppress verbose output"""
        # Store original Streamlit functions
        original_info = st.info
        original_success = st.success
        original_warning = st.warning
        original_spinner = st.spinner
        
        # Create silent versions
        def silent_info(*args, **kwargs):
            pass
        def silent_success(*args, **kwargs):
            pass
        def silent_warning(*args, **kwargs):
            pass
        def silent_spinner(text):
            from contextlib import nullcontext
            return nullcontext()
        
        try:
            # Replace with silent versions
            st.info = silent_info
            st.success = silent_success
            st.warning = silent_warning
            st.spinner = silent_spinner
            
            # Call parent method
            return super().verify_candidate_with_urls(candidate_info, resume_text, embedded_links)
        
        finally:
            # Restore original functions
            st.info = original_info
            st.success = original_success
            st.warning = original_warning
            st.spinner = original_spinner

# Load environment variables - DISABLED
# load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Resume Screening & Ranking System",
    page_icon="📄",
    layout="wide"
)

# Create folders if they don't exist
if not os.path.exists("uploaded_resumes"):
    os.makedirs("uploaded_resumes")

if not os.path.exists("cache"):
    os.makedirs("cache")

# Hardcoded Groq API keys - no .env file needed
GROQ_API_KEYS = [
    "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"
    # Add more API keys here if you have them for better rate limiting
    # "your_second_api_key",
    # "your_third_api_key"
]

# GitHub API token for higher rate limits
GITHUB_TOKEN = "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"  # GitHub personal access token

# Remove the warning since we have hardcoded keys
# if not GROQ_API_KEYS:
#     st.sidebar.warning("⚠️ No Groq API keys found. Please add them to your .env file as GROQ_API_KEY_1, GROQ_API_KEY_2, etc.")

# API key tracking and rate limiting
class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.key_usage = {key: {"last_used": datetime.now() - timedelta(minutes=2), "calls": 0} for key in api_keys}
        self.lock = threading.Lock()
    
    def get_next_available_key(self):
        """Get the next available API key based on usage and time since last call."""
        if not self.api_keys:
            return None
            
        with self.lock:
            # Sort keys by last used time (oldest first) and usage count
            sorted_keys = sorted(
                self.api_keys,
                key=lambda k: (
                    (datetime.now() - self.key_usage[k]["last_used"]).total_seconds(),
                    -self.key_usage[k]["calls"]  # Negative to prioritize keys with fewer calls
                ),
                reverse=True
            )
            
            # Get the key that has been waiting the longest and used least
            selected_key = sorted_keys[0]
            
            # Update usage tracking
            self.key_usage[selected_key]["last_used"] = datetime.now()
            self.key_usage[selected_key]["calls"] += 1
            
            return selected_key
    
    def mark_key_rate_limited(self, key):
        """Mark a key as rate limited to avoid using it for a cooling period."""
        with self.lock:
            if key in self.key_usage:
                # Set last_used to now + 60 seconds to implement a cooling period
                self.key_usage[key]["last_used"] = datetime.now() + timedelta(seconds=60)
    
    def get_key_status(self):
        """Get status information about all keys."""
        status = {}
        for key in self.api_keys:
            time_since_last_use = (datetime.now() - self.key_usage[key]["last_used"]).total_seconds()
            status[key[:8]] = {
                "calls": self.key_usage[key]["calls"],
                "seconds_since_last_use": time_since_last_use if time_since_last_use > 0 else "cooling",
                "available": time_since_last_use > 0
            }
        return status

# Initialize API key manager
api_key_manager = APIKeyManager(GROQ_API_KEYS)

# Request queue system
class APIRequestQueue:
    def __init__(self):
        self.request_queue = queue.Queue()
        self.response_dict = {}
        self.worker_thread = None
        self.is_processing = False
    
    def start_worker(self):
        """Start the worker thread to process the queue."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.is_processing = True
            self.worker_thread = threading.Thread(target=self._process_queue)
            self.worker_thread.daemon = True
            self.worker_thread.start()
    
    def _process_queue(self):
        """Process queued API requests."""
        while self.is_processing:
            try:
                if self.request_queue.empty():
                    time.sleep(0.5)
                    continue
                
                request_id, prompt, model, max_retries = self.request_queue.get()
                
                # Process the request
                for attempt in range(max_retries):
                    # Use exponential backoff for retries
                    if attempt > 0:
                        wait_time = 2 ** attempt  # 2, 4, 8, 16, 32... seconds
                        time.sleep(wait_time)
                    
                    # Get next available API key
                    api_key = api_key_manager.get_next_available_key()
                    if not api_key:
                        self.response_dict[request_id] = {"error": "No API keys available"}
                        break
                    
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "messages": [{"role": "user", "content": prompt}],
                        "model": model,
                        "temperature": 0.1,
                        "max_tokens": 4096
                    }
                    
                    try:
                        response = requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                        response.raise_for_status()
                        self.response_dict[request_id] = response.json()["choices"][0]["message"]["content"]
                        break
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 429:  # Rate limit
                            api_key_manager.mark_key_rate_limited(api_key)
                            if attempt == max_retries - 1:
                                self.response_dict[request_id] = {"error": f"Rate limit exceeded after {max_retries} attempts"}
                        else:
                            self.response_dict[request_id] = {"error": f"HTTP Error: {str(e)}"}
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            self.response_dict[request_id] = {"error": str(e)}
                
                self.request_queue.task_done()
                
            except Exception as e:
                st.error(f"Error in queue worker: {str(e)}")
                time.sleep(1)
    
    def add_request(self, prompt, model="llama3-70b-8192", max_retries=5):
        """Add a request to the queue and return a request ID."""
        request_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()
        self.request_queue.put((request_id, prompt, model, max_retries))
        return request_id
    
    def get_response(self, request_id, timeout=None):
        """Get a response for a given request ID, with optional timeout."""
        start_time = time.time()
        while request_id not in self.response_dict:
            if timeout and time.time() - start_time > timeout:
                return {"error": "Request timed out"}
            time.sleep(0.5)
        
        response = self.response_dict.pop(request_id)
        return response
    
    def stop_worker(self):
        """Stop the worker thread."""
        self.is_processing = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1)

# Initialize request queue
api_request_queue = APIRequestQueue()
api_request_queue.start_worker()

# Cache system for API responses
def get_cache_key(prompt, model):
    """Generate a cache key based on prompt and model."""
    return hashlib.md5(f"{prompt}:{model}".encode()).hexdigest()

def get_cached_response(prompt, model="llama3-70b-8192"):
    """Try to get a response from cache."""
    cache_key = get_cache_key(prompt, model)
    cache_file = os.path.join("cache", f"{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            return cached_data
        except:
            return None
    return None

def cache_response(prompt, response, model="llama3-70b-8192"):
    """Cache an API response."""
    cache_key = get_cache_key(prompt, model)
    cache_file = os.path.join("cache", f"{cache_key}.pkl")
    
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(response, f)
    except Exception as e:
        st.warning(f"Failed to cache response: {str(e)}")

# Add this helper function to sanitize and parse JSON responses
def parse_llm_response(response_text):
    """Safely parse LLM response and ensure it's valid JSON."""
    try:
        # First try direct JSON parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Try to extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                return json.loads(json_match.group(1))
            
            # If no code blocks, try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group(0))
            
            # If still no valid JSON, return the response as plain text
            return {"error": "Could not parse response as JSON", "text": response_text}
        except Exception as e:
            return {"error": f"Error parsing response: {str(e)}", "text": response_text}

# Improved function to call Groq API with caching, retry, and queuing
def call_groq_api(prompt, model="llama3-70b-8192", max_retries=5, use_cache=True):
    """
    Call the Groq API with caching, retries, exponential backoff, and queue management.
    """
    if not GROQ_API_KEYS:
        return {"error": "No Groq API keys available"}
    
    # Check cache first if enabled
    if use_cache:
        cached_response = get_cached_response(prompt, model)
        if cached_response is not None:
            return cached_response
    
    # Add request to queue
    request_id = api_request_queue.add_request(prompt, model, max_retries)
    
    # Wait for response with a reasonable timeout (30 seconds per retry)
    timeout = 30 * max_retries
    response = api_request_queue.get_response(request_id, timeout)
    
    # Cache successful responses
    if use_cache and isinstance(response, str):
        cache_response(prompt, response, model)
    
    return response

# Function to extract text from PDF with better structure preservation
def extract_text_from_pdf(pdf_file):
    """Extract text and hyperlinks from a PDF file."""
    text = ""
    links = []
    try:
        if isinstance(pdf_file, (str, Path)):
            # If pdf_file is a filepath
            pdf_document = fitz.open(pdf_file)
        else:
            # If pdf_file is a file object
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Extract text blocks to preserve structure
            blocks = page.get_text("blocks")
            for block in blocks:
                text += block[4] + "\n"
            
            # Extract links from the page
            link_objects = page.get_links()
            for link in link_objects:
                if link.get("uri"):
                    links.append(link["uri"])
            
            # Also extract URLs from annotations
            for annot in page.annots():
                if annot.type[0] == 2:  # Link annotation
                    if annot.info.get("content"):
                        # Try to extract URL from content
                        url_match = re.search(r'https?://[^\s<>"{}|\\^`\[\]]+', annot.info["content"])
                        if url_match:
                            links.append(url_match.group())
                    # Check if annotation has a URI
                    if hasattr(annot, 'uri') and annot.uri:
                        links.append(annot.uri)
                
            # Add extra line break between pages
            text += "\n"
            
        # Clean up text - remove excessive newlines and spaces
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove duplicate links
        links = list(set(links))
        
        # Return both text and links
        return {"text": text, "links": links}
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return {"text": "", "links": []}

# Function to handle zip file upload
def handle_zip_upload(zip_file):
    """Extract resume texts and links from a zip file."""
    resume_data = {}
    
    try:
        with zipfile.ZipFile(zip_file) as z:
            for file_name in z.namelist():
                if file_name.endswith('.pdf') and not file_name.startswith('__MACOSX'):
                    with z.open(file_name) as f:
                        content = BytesIO(f.read())
                        try:
                            extracted_data = extract_text_from_pdf(content)
                            resume_data[file_name] = extracted_data
                        except Exception as e:
                            st.error(f"Error processing {file_name}: {e}")
    except Exception as e:
        st.error(f"Error processing zip file: {e}")
    
    return resume_data

# Enhanced job description analysis using direct Groq API call
def analyze_job_description(job_description_text):
    """Analyze job description to extract key requirements and skills."""
    # Convert to lowercase for consistent processing
    job_description_text = job_description_text.strip()
    
    if not job_description_text:
        st.error("❌ Please enter a job description")
        return {}
    
    prompt = f"""
    You are an expert recruiter and job analyst. Analyze this job description written in natural language and extract all relevant requirements. The description might be informal, conversational, or unstructured - your job is to understand the context and extract meaningful requirements.
    
    JOB DESCRIPTION:
    {job_description_text}

    Based on the context and meaning, extract:

    1. Technical Skills (programming languages, frameworks, tools, technologies mentioned or implied)
    2. Soft Skills (communication, leadership, teamwork, etc. - even if informally mentioned)
    3. Certifications (any certifications that would be valuable, even if not explicitly required)
    4. Education (degrees, fields of study, or educational background mentioned or implied)
    5. Experience (years of experience, types of projects, industry experience)
    6. Responsibilities (what the person will actually do, even if casually mentioned)
    7. Keywords (important terms, technologies, methodologies, industry-specific words)

    IMPORTANT: 
    - Understand the CONTEXT. If they say "build websites", infer HTML, CSS, JavaScript
    - If they mention "data analysis", infer Python, SQL, Excel, etc.
    - If they say "team player", extract teamwork as a soft skill
    - Extract both EXPLICIT and IMPLIED requirements
    - Be thorough - it's better to extract more than to miss important requirements

    Format as JSON:
    {{
        "technical_skills": ["skill1", "skill2", ...],
        "soft_skills": ["skill1", "skill2", ...],
        "certifications": ["cert1", "cert2", ...],
        "education": ["requirement1", "requirement2", ...],
        "experience": ["exp1", "exp2", ...],
        "responsibilities": ["resp1", "resp2", ...],
        "keywords": ["keyword1", "keyword2", ...]
    }}
    """
    
    try:
        response = call_groq_api(prompt, model="llama3-70b-8192", max_retries=3)
        
        if isinstance(response, dict) and "error" in response:
            st.error(f"Error analyzing job description: {response['error']}")
            return {}
        
        # Clean and parse the response
        json_str = response
        if isinstance(response, str):
            # Try to extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code blocks, try to find JSON object
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    json_str = json_match.group(0)
        
        # Clean up the JSON string
        json_str = json_str.strip()
        if not json_str.startswith("{"):
            json_str = json_str[json_str.find("{"):]
        if not json_str.endswith("}"):
            json_str = json_str[:json_str.rfind("}") + 1]
        
        try:
            job_req = json.loads(json_str)
            
            # Validate and clean the extracted requirements
            required_keys = ["technical_skills", "soft_skills", "certifications", 
                           "education", "experience", "responsibilities", "keywords"]
            
            # Ensure all required keys exist and are lists
            for key in required_keys:
                if key not in job_req:
                    job_req[key] = []
                elif not isinstance(job_req[key], list):
                    job_req[key] = [str(job_req[key])]
                
                # Clean and validate each item in the list
                cleaned_items = []
                for item in job_req[key]:
                    if item and isinstance(item, str):
                        cleaned_item = item.strip()
                        if cleaned_item and cleaned_item not in cleaned_items:
                            cleaned_items.append(cleaned_item)
                job_req[key] = cleaned_items
            
            # Show what was extracted
            st.success("✅ Understood the job requirements!")
            with st.expander("See what I extracted from your description", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Technical Requirements:**")
                    for skill in job_req.get("technical_skills", [])[:10]:
                        st.write(f"• {skill}")
                    if len(job_req.get("technical_skills", [])) > 10:
                        st.write(f"• ... and {len(job_req['technical_skills']) - 10} more")
                        
                with col2:
                    st.write("**Experience & Soft Skills:**")
                    for exp in job_req.get("experience", [])[:5]:
                        st.write(f"• {exp}")
                    for skill in job_req.get("soft_skills", [])[:5]:
                        st.write(f"• {skill}")
            
            return job_req
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing job requirements: {str(e)}")
            st.error("Raw response from API:")
            st.code(response)
            return {}
            
    except Exception as e:
        st.error(f"Error analyzing job description: {str(e)}")
        return {}

# Function to create embeddings and build a FAISS index
def build_faiss_index(resume_data, job_requirements):
    """Create embeddings and build a FAISS index for efficient searching."""
    # Combine job requirements into a single text for vectorization
    job_req_text = ""
    for category, items in job_requirements.items():
        if isinstance(items, list):
            job_req_text += f"{category}: {', '.join(items)}\n"
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1024,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Create a list of all documents (resumes + job description)
    documents = [data["text"] for data in resume_data.values()]
    documents.append(job_req_text)  # Add job description as the last document
    
    # Fit and transform to create document vectors
    document_vectors = vectorizer.fit_transform(documents).astype('float32').toarray()
    
    # Create FAISS index (using flat index for maximum accuracy)
    dimension = document_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product index for cosine similarity
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(document_vectors)
    
    # Add document vectors to index
    index.add(document_vectors[:-1])  # Add all except the job description
    
    # Return the index, vectorizer, and job description vector
    return index, vectorizer, document_vectors[-1]  # Job description is the last vector

# Function to search FAISS index for relevant resumes
def search_relevant_resumes(index, job_vector, resume_data, k=None):
    """Search the FAISS index for resumes most similar to the job description."""
    if k is None:
        k = len(resume_data)  # Search for all resumes
    
    # Reshape query vector and normalize
    query_vector = job_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)
    
    # Search the index
    D, I = index.search(query_vector, k)
    
    # Create result mapping
    results = []
    filenames = list(resume_data.keys())
    
    for i, (idx, score) in enumerate(zip(I[0], D[0])):
        if idx < len(filenames):  # Ensure valid index
            results.append({
                "filename": filenames[idx],
                "similarity_score": float(score),
                "rank": i + 1
            })
    
    return results

# Function to extract resume information using LLM
def extract_resume_info(resume_text):
    """Extract structured information from resume text using Groq API."""
    prompt = f"""
    Extract the following information from this resume. If information is not found, return empty string or appropriate default:

    RESUME TEXT:
    {resume_text[:5000]}  # Limit text to avoid token issues

    Please extract:
    1. Full name
    2. Email address
    3. Phone number
    4. Education details (degree, institution, graduation year, GPA/CGPA if available)
    5. Technical skills
    6. Work experience (company names, roles, duration)
    7. Projects
    8. Certifications
    9. Regions/Location information

    Format your response as a structured JSON with the following keys:
    "name", "email", "phone", "education", "skills", "experience", "projects", "certifications", "region"
    
    For lists (education, skills, experience, projects, certifications), return arrays of strings.
    """
    
    try:
        response = call_groq_api(prompt)
        
        # Parse the JSON response
        if isinstance(response, dict) and "error" in response:
            st.error(f"Error extracting resume information: {response['error']}")
            return {}
            
        # Extract JSON from potentially mixed text response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
            
        # Remove any non-JSON content before or after the JSON object
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error parsing resume information: {e}")
        return {}

# Function to match resume against job requirements using RAG approach
def match_resume_to_job(resume_info, job_requirements):
    """Match resume against job requirements using LLM with RAG approach."""
    # Define required keys at the start of the function
    required_score_keys = ["technical_skills", "soft_skills", "education", 
                          "experience", "certifications", "overall_match"]
    required_explanation_keys = ["technical_skills", "soft_skills", "education", 
                               "experience", "certifications"]

    # Default response structure
    default_response = {
        "scores": {k: 0 for k in required_score_keys},
        "explanations": {k: "No analysis available" for k in required_explanation_keys},
        "skill_matches": [],
        "skill_gaps": [],
        "recommendation": "No recommendation available"
    }

    try:
        # Create the prompt with explicit formatting instructions
        prompt = f"""
        You are an expert recruiter. Match this candidate's profile against the job requirements.
        
        IMPORTANT: Consider the CONTEXT and MEANING, not just exact keyword matches:
        - If job needs "web development" and candidate has "React, Node.js" - that's a match!
        - Look at their PROJECTS to understand what they can do
        - Experience in similar technologies counts (e.g., Java experience helps for C# roles)
        - Consider transferable skills
        
        Candidate's Resume:
        Name: {resume_info.get('name', 'Unknown')}
        Skills: {', '.join(resume_info.get('skills', []))}
        Experience: {json.dumps(resume_info.get('experience', []))}
        Projects: {json.dumps(resume_info.get('projects', []))}
        Education: {json.dumps(resume_info.get('education', []))}
        Certifications: {json.dumps(resume_info.get('certifications', []))}
        
        Job Requirements:
        {json.dumps(job_requirements, indent=2)}
        
        Analyze how well this candidate matches. Consider:
        1. Direct skill matches
        2. Related/transferable skills
        3. Project experience that demonstrates required abilities
        4. Overall potential to succeed in this role
        
        Provide scores (0-100) and explanations in this JSON format:
        {{
            "scores": {{
                "technical_skills": <0-100>,
                "soft_skills": <0-100>,
                "education": <0-100>,
                "experience": <0-100>,
                "certifications": <0-100>,
                "overall_match": <0-100>
            }},
            "explanations": {{
                "technical_skills": "explanation",
                "soft_skills": "explanation",
                "education": "explanation",
                "experience": "explanation",
                "certifications": "explanation"
            }},
            "skill_matches": ["matched skill 1", "matched skill 2"],
            "skill_gaps": ["missing skill 1", "missing skill 2"],
            "recommendation": "overall hiring recommendation"
        }}
        
        Be fair and consider potential, not just exact matches.
        """
        
        # Call Groq API with retries
        response = call_groq_api(prompt, max_retries=3)
        
        if isinstance(response, dict) and "error" in response:
            st.error(f"Error matching resume to job: {response['error']}")
            return default_response
        
        # Clean and parse the response
        try:
            # First try direct JSON parsing
            result = json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to extract JSON from markdown code blocks if present
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # If no code blocks, try to find JSON object
                    json_match = re.search(r'\{[\s\S]*\}', response)
                    if json_match:
                        # Clean the JSON string
                        json_str = json_match.group(0)
                        # Remove any invalid control characters
                        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
                        result = json.loads(json_str)
                    else:
                        return default_response
            except Exception:
                return default_response
        
        # Validate and normalize the result structure
        if not isinstance(result, dict):
            return default_response
        
        # Ensure all required sections exist
        if "scores" not in result:
            result["scores"] = {}
        if "explanations" not in result:
            result["explanations"] = {}
        if "skill_matches" not in result:
            result["skill_matches"] = []
        if "skill_gaps" not in result:
            result["skill_gaps"] = []
        if "recommendation" not in result:
            result["recommendation"] = "No recommendation available"
        
        # Validate and normalize scores
        for key in required_score_keys:
            try:
                score = float(result["scores"].get(key, 0))
                result["scores"][key] = max(0, min(100, score))  # Ensure score is between 0 and 100
            except (ValueError, TypeError):
                result["scores"][key] = 0
        
        # Validate and normalize explanations
        for key in required_explanation_keys:
            if key not in result["explanations"] or not isinstance(result["explanations"][key], str):
                result["explanations"][key] = "No analysis available"
        
        # Validate skill lists
        if not isinstance(result["skill_matches"], list):
            result["skill_matches"] = []
        if not isinstance(result["skill_gaps"], list):
            result["skill_gaps"] = []
        
        # Ensure recommendation is a string
        if not isinstance(result["recommendation"], str):
            result["recommendation"] = "No recommendation available"
        
        return result
        
    except Exception as e:
        st.error(f"Error in resume matching: {str(e)}")
        return default_response

# Function to create a download link for CSV file
def get_csv_download_link(df, filename="ranked_candidates.csv"):
    """Create a download link for a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn" style="background-color:#4CAF50;color:white;padding:8px 12px;text-decoration:none;border-radius:4px;">Download Ranked Candidates CSV</a>'
    return href

# Function to filter candidates by region
def filter_by_region(results, region=None):
    """Filter candidates by region if specified."""
    if not region or region.lower() == 'all':
        return results
    
    filtered_results = []
    for result in results:
        if 'resume_info' in result and 'region' in result['resume_info']:
            candidate_region = result['resume_info'].get('region', '').lower()
            if region.lower() in candidate_region:
                filtered_results.append(result)
    
    return filtered_results

# Modify the chat input handler to only use session state
def handle_chat_input(i, resume_info, candidate_id):
    """Handle chat input submission."""
    # Get the question from the session state
    user_question = st.session_state.get(f"question_{i}")
    if user_question:
        # Prepare context for the LLM
        context = f"""
        Resume Information:
        Name: {resume_info.get('name', 'Unknown')}
        Education: {', '.join(resume_info.get('education', []))}
        Experience: {', '.join(resume_info.get('experience', []))}
        Skills: {', '.join(resume_info.get('skills', []))}
        Projects: {', '.join(resume_info.get('projects', []))}
        Certifications: {', '.join(resume_info.get('certifications', []))}
        
        Question: {user_question}
        
        Please provide a natural language response. Do not format as JSON.
        Base your answer only on the resume information provided above.
        If the information is not available in the resume, please say so.
        """
        
        # Call Groq API for response
        prompt = f"Based on the following resume information, please answer this question: {context}"
        response = call_groq_api(prompt)
        
        if isinstance(response, dict) and "error" in response:
            st.error(f"Error: {response['error']}")
        else:
            # Just use the response directly as text
            answer = str(response).strip()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                # Initialize session chat history if not exists
                if f"session_chat_{i}" not in st.session_state:
                    st.session_state[f"session_chat_{i}"] = []
                
                # Add to session chat history
                st.session_state[f"session_chat_{i}"].append({
                    "timestamp": timestamp,
                    "question": user_question,
                    "answer": answer
                })
                
                # Clear the input
                st.session_state[f"question_{i}"] = ""
                
            except Exception as e:
                st.error(f"Error handling chat: {str(e)}")

# Modify the resume processing section in main() to use concurrent processing
def process_resume_batch(batch, resume_data, job_requirements):
    """Process a batch of resumes concurrently."""
    results = []
    verifier = SilentCandidateVerifier(github_token=GITHUB_TOKEN)  # Pass GitHub token
    
    for result in batch:
        try:
            filename = result["filename"]
            resume_text = resume_data[filename]["text"]
            resume_links = resume_data[filename]["links"]
            
            # Extract structured information from resume
            resume_info = extract_resume_info(resume_text)
            
            # Enhanced candidate verification with web scraping
            verification_result = None
            
            try:
                # Initialize custom verifier with minimal output
                custom_verifier = SilentCandidateVerifier(github_token=GITHUB_TOKEN)  # Pass GitHub token
                
                # Call verification with suppressed output
                verification_result = custom_verifier.verify_candidate_with_urls(
                    candidate_info=resume_info,
                    resume_text=resume_text,
                    embedded_links=resume_links
                )
                
            except Exception as verify_error:
                verification_result = None
            
            # Add verification data to resume_info
            if verification_result and not verification_result.get('error'):
                resume_info.update({
                    'verified_email': verification_result.get('email_verification', {}).get('is_valid', False),
                    'domain_reputation': verification_result.get('email_verification', {}).get('domain_reputation', 'unknown'),
                    'social_profiles': verification_result.get('social_profiles', {}),
                    'github_profile': verification_result.get('github_data', {}),
                    'linkedin_profile': verification_result.get('linkedin_data', {}),
                    'authenticity_score': verification_result.get('authenticity_score', 0)
                })
            else:
                # Set default values if verification failed
                resume_info.update({
                    'verified_email': False,
                    'domain_reputation': 'unknown',
                    'social_profiles': {},
                    'github_profile': {},
                    'linkedin_profile': {},
                    'authenticity_score': 0
                })
            
            # Match resume against job requirements
            match_result = match_resume_to_job(resume_info, job_requirements)
            
            # Add to results with all verification data
            results.append({
                "filename": filename,
                "similarity_score": result["similarity_score"],
                "resume_info": resume_info,
                "match_result": match_result,
                "verification_result": verification_result,
                "profiles": {
                    "github": verification_result.get('profiles', {}).get('github', {}) if verification_result else {},
                    "linkedin": verification_result.get('profiles', {}).get('linkedin', {}) if verification_result else {}
                },
                "scraped_content": verification_result.get('scraped_content', {}) if verification_result else {}
            })
        except Exception as e:
            st.error(f"Error processing resume {filename}: {str(e)}")
            continue
    
    return results

# Function to call Ollama API for chatbot - REMOVED in favor of Groq API
# def call_ollama_api(prompt, model="llama3.1:8b", max_retries=3):
#     """
#     Call the Ollama API for chatbot functionality.
#     """
#     import requests
    
#     for attempt in range(max_retries):
#         try:
#             response = requests.post(
#                 "http://localhost:11434/api/generate",
#                 json={
#                     "model": model,
#                     "prompt": prompt,
#                     "stream": False
#                 },
#                 timeout=30
#             )
#             response.raise_for_status()
#             return response.json().get("response", "")
#         except requests.exceptions.RequestException as e:
#             if attempt == max_retries - 1:
#                 return f"Error connecting to Ollama: {str(e)}"
#             time.sleep(2)  # Wait before retry
    
#     return "Error: Could not connect to Ollama after multiple attempts"

# Remove Ollama API function and replace with Groq API for chat
def generate_chat_response(user_input: str, context: Dict[str, Any]) -> str:
    """
    Generate a response to a user's question about a candidate using Groq API with comprehensive context.
    """
    
    # Extract all candidate information
    resume_info = context.get('resume_info', {})
    match_result = context.get('match_result', {})
    scraped_content = context.get('scraped_content', {})
    profiles = context.get('profiles', {})
    verification_result = context.get('verification_result', {})
    
    # Build comprehensive context
    context_parts = []
    
    # Basic candidate information
    context_parts.append("=== CANDIDATE PROFILE ===")
    context_parts.append(f"Name: {resume_info.get('name', 'Unknown')}")
    context_parts.append(f"Email: {resume_info.get('email', 'Not provided')}")
    context_parts.append(f"Phone: {resume_info.get('phone', 'Not provided')}")
    context_parts.append(f"Location/Region: {resume_info.get('region', 'Not specified')}")
    
    # Skills and technical expertise
    context_parts.append("\n=== TECHNICAL SKILLS ===")
    skills = resume_info.get('skills', [])
    if skills:
        context_parts.append(f"Skills: {', '.join(skills)}")
    else:
        context_parts.append("Skills: Not specified in resume")
    
    # Experience details
    context_parts.append("\n=== WORK EXPERIENCE ===")
    experience = resume_info.get('experience', [])
    if experience:
        for i, exp in enumerate(experience, 1):
            context_parts.append(f"{i}. {exp}")
    else:
        context_parts.append("Experience: Not detailed in resume")
    
    # Education background
    context_parts.append("\n=== EDUCATION ===")
    education = resume_info.get('education', [])
    if education:
        for i, edu in enumerate(education, 1):
            context_parts.append(f"{i}. {edu}")
    else:
        context_parts.append("Education: Not specified in resume")
    
    # Projects
    context_parts.append("\n=== PROJECTS ===")
    projects = resume_info.get('projects', [])
    if projects:
        for i, project in enumerate(projects, 1):
            context_parts.append(f"{i}. {project}")
    else:
        context_parts.append("Projects: None mentioned in resume")
    
    # Certifications
    context_parts.append("\n=== CERTIFICATIONS ===")
    certifications = resume_info.get('certifications', [])
    if certifications:
        for i, cert in enumerate(certifications, 1):
            context_parts.append(f"{i}. {cert}")
    else:
        context_parts.append("Certifications: None mentioned")
    
    # Match analysis
    context_parts.append("\n=== JOB MATCH ANALYSIS ===")
    scores = match_result.get('scores', {})
    context_parts.append(f"Overall Match Score: {scores.get('overall_match', 0):.1f}%")
    context_parts.append(f"Technical Skills Match: {scores.get('technical_skills', 0):.1f}%")
    context_parts.append(f"Experience Match: {scores.get('experience', 0):.1f}%")
    context_parts.append(f"Education Match: {scores.get('education', 0):.1f}%")
    context_parts.append(f"Soft Skills Match: {scores.get('soft_skills', 0):.1f}%")
    context_parts.append(f"Certifications Match: {scores.get('certifications', 0):.1f}%")
    
    # Skill matches and gaps
    skill_matches = match_result.get('skill_matches', [])
    skill_gaps = match_result.get('skill_gaps', [])
    if skill_matches:
        context_parts.append(f"Matching Skills: {', '.join(skill_matches)}")
    if skill_gaps:
        context_parts.append(f"Skill Gaps: {', '.join(skill_gaps)}")
    
    # Recommendation
    recommendation = match_result.get('recommendation', '')
    if recommendation:
        context_parts.append(f"AI Recommendation: {recommendation}")
    
    # Detailed match explanations
    explanations = match_result.get('explanations', {})
    if explanations:
        context_parts.append("\n=== DETAILED MATCH EXPLANATIONS ===")
        if explanations.get('technical_skills'):
            context_parts.append(f"Technical Skills Analysis: {explanations['technical_skills']}")
        if explanations.get('experience'):
            context_parts.append(f"Experience Analysis: {explanations['experience']}")
        if explanations.get('education'):
            context_parts.append(f"Education Analysis: {explanations['education']}")
        if explanations.get('soft_skills'):
            context_parts.append(f"Soft Skills Analysis: {explanations['soft_skills']}")
        if explanations.get('certifications'):
            context_parts.append(f"Certifications Analysis: {explanations['certifications']}")
    
    # GitHub profile information
    github_data = profiles.get('github', {})
    if github_data and github_data.get('found'):
        print("\n=== DEBUG: GitHub Data ===")
        print(f"GitHub Data: {json.dumps(github_data, indent=2)}")
        context_parts.append("\n=== GITHUB PROFILE ===")
        if github_data.get('profile_url'):
            context_parts.append(f"GitHub URL: {github_data['profile_url']}")
        if github_data.get('bio'):
            context_parts.append(f"Bio: {github_data['bio']}")
        if github_data.get('public_repos'):
            context_parts.append(f"Public Repositories: {github_data['public_repos']}")
        if github_data.get('followers'):
            context_parts.append(f"Followers: {github_data['followers']}")
        if github_data.get('following'):
            context_parts.append(f"Following: {github_data['following']}")
        if github_data.get('company'):
            context_parts.append(f"Company: {github_data['company']}")
        if github_data.get('location'):
            context_parts.append(f"Location: {github_data['location']}")
        if github_data.get('blog'):
            context_parts.append(f"Website/Blog: {github_data['blog']}")
        if github_data.get("repositories"):
            print("\n=== DEBUG: GitHub Repositories ===")
            print(f"Repositories: {json.dumps(github_data['repositories'], indent=2)}")
            context_parts.append("\nNotable Repositories:")
            repos = github_data.get("repositories", [])
            if isinstance(repos, list):
                for repo in repos[:3]:  # Show only top 3 repositories
                    if isinstance(repo, dict):
                        repo_info = []
                        if repo.get('name'):
                            repo_info.append(f"**{repo['name']}**")
                        if repo.get('description'):
                            repo_info.append(f"Description: {repo['description']}")
                        if repo.get('language'):
                            repo_info.append(f"Language: {repo['language']}")
                        if repo.get('stars'):
                            repo_info.append(f"Stars: {repo['stars']}")
                        if repo.get('forks'):
                            repo_info.append(f"Forks: {repo['forks']}")
                        if repo.get('url'):
                            repo_info.append(f"URL: {repo['url']}")
                        context_parts.append("• " + " | ".join(repo_info))
                    else:
                        context_parts.append(f"• {repo}")
        if github_data.get('contributions'):
            context_parts.append(f"Contributions: {github_data['contributions']}")
        if github_data.get('languages'):
            context_parts.append(f"Programming Languages Used: {', '.join(github_data['languages'])}")
    
        context_parts.append("---")
    
    # LinkedIn profile information
    linkedin_data = profiles.get('linkedin', {})
    if linkedin_data and linkedin_data.get('found'):
        print("\n=== DEBUG: LinkedIn Data ===")
        print(f"LinkedIn Data: {json.dumps(linkedin_data, indent=2)}")
        context_parts.append("\n=== LINKEDIN PROFILE ===")
        if linkedin_data.get('verified_urls'):
            urls = linkedin_data['verified_urls']
            if isinstance(urls, list) and urls:
                context_parts.append(f"LinkedIn URL: {urls[0]}")
                linkedin_url = urls[0]
                username = linkedin_url.split('/')[-1]  # Extract username from URL
                
                # LinkedIn recent activity API call
                try:
                    print("\n=== DEBUG: LinkedIn Recent Activity API Call ===")
                    activity_url = "https://linkedin-data-api.p.rapidapi.com/profiles/recent-activity/all"
                    activity_payload = {
                        "username": username,
                        "page": 1,
                        "limit": 1  # Get only 1 most recent activity
                    }
                    activity_headers = {
                        "x-rapidapi-key": "512175d384mshb7894ebd80ddda4p11a2c4jsn7fbc012751c2",
                        "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com",
                        "Content-Type": "application/json"
                    }
                    
                    print(f"Making request to: {activity_url}")
                    print(f"With payload: {json.dumps(activity_payload, indent=2)}")
                    
                    activity_response = requests.post(
                        activity_url,
                        json=activity_payload,
                        headers=activity_headers
                    )
                    
                    print(f"Response status code: {activity_response.status_code}")
                    print(f"Response content: {activity_response.text}")
                    
                    if activity_response.status_code == 200:
                        activity_data = activity_response.json()
                        if activity_data and isinstance(activity_data, dict):
                            context_parts.append("\nRecent Activity:")
                            if 'activities' in activity_data and activity_data['activities']:
                                activity = activity_data['activities'][0]  # Get the first activity
                                with st.container():
                                    st.markdown("---")
                                    activity_type = activity.get('type', 'Unknown')
                                    activity_date = activity.get('date', 'Unknown date')
                                    activity_text = activity.get('text', 'No text available')
                                    activity_likes = activity.get('likes', 0)
                                    activity_comments = activity.get('comments', 0)
                                    
                                    # Display activity type with appropriate emoji
                                    activity_emoji = {
                                        'post': '📝',
                                        'article': '📰',
                                        'comment': '💬',
                                        'like': '👍',
                                        'share': '🔄',
                                        'job': '💼',
                                        'connection': '🤝',
                                        'certification': '🏆',
                                        'default': '📌'
                                    }.get(activity_type.lower(), '📌')
                                    
                                    st.markdown(f"{activity_emoji} **{activity_type.title()}** on {activity_date}")
                                    st.markdown(f"{activity_text[:200]}..." if len(activity_text) > 200 else activity_text)
                                    
                                    # Show engagement metrics if available
                                    if activity_likes > 0 or activity_comments > 0:
                                        st.markdown(f"👍 {activity_likes} likes | 💬 {activity_comments} comments")
                                    
                                    # Show additional activity details if available
                                    if activity.get('company'):
                                        st.markdown(f"🏢 Company: {activity['company']}")
                                    if activity.get('title'):
                                        st.markdown(f"📋 Title: {activity['title']}")
                                    if activity.get('url'):
                                        st.markdown(f"[View Activity]({activity['url']})")
                except Exception as e:
                    print(f"Error in recent activity API call: {str(e)}")
                    context_parts.append(f"Note: Could not fetch recent activity: {str(e)}")
            elif isinstance(urls, str):
                context_parts.append(f"LinkedIn URL: {urls}")
    
    # Add LinkedIn profile details
    if linkedin_data.get('headline'):
        context_parts.append(f"Headline: {linkedin_data['headline']}")
    if linkedin_data.get('summary'):
        context_parts.append(f"Summary: {linkedin_data['summary']}")
    if linkedin_data.get('location'):
        context_parts.append(f"Location: {linkedin_data['location']}")
    if linkedin_data.get('industry'):
        context_parts.append(f"Industry: {linkedin_data['industry']}")
    if linkedin_data.get('connections'):
        context_parts.append(f"Connections: {linkedin_data['connections']}")
    
    context_parts.append("---")
    
    # Other web presence
    if scraped_content:
        context_parts.append("\n=== OTHER WEB PRESENCE ===")
        for url, content in scraped_content.items():
            if content and isinstance(content, dict) and content.get('success'):
                platform = content.get('platform', 'Website')
                context_parts.append(f"{platform} ({url}):")
                if content.get('title'):
                    context_parts.append(f"  Title: {content['title']}")
                if content.get('description'):
                    context_parts.append(f"  Description: {content['description']}")
                if content.get('skills'):
                    skills = content['skills']
                    if isinstance(skills, list):
                        context_parts.append(f"  Skills Found: {', '.join(skills)}")
                    else:
                        context_parts.append(f"  Skills Found: {skills}")
                if content.get('technologies'):
                    tech = content['technologies']
                    if isinstance(tech, list):
                        context_parts.append(f"  Technologies: {', '.join(tech)}")
                    else:
                        context_parts.append(f"  Technologies: {tech}")
                if content.get('projects'):
                    projects = content['projects']
                    if isinstance(projects, list):
                        context_parts.append(f"  Projects Found: {', '.join(projects)}")
                    else:
                        context_parts.append(f"  Projects Found: {projects}")
                if content.get('experience'):
                    context_parts.append(f"  Experience Found: {content['experience']}")
                if content.get('education'):
                    context_parts.append(f"  Education Found: {content['education']}")
    
    # Verification and authenticity
    if verification_result and not verification_result.get('error'):
        context_parts.append("\n=== VERIFICATION STATUS ===")
        
        # Email verification details
        if resume_info.get('verified_email') is not None:
            context_parts.append(f"Email Verified: {'Yes' if resume_info.get('verified_email') else 'No'}")
        if resume_info.get('domain_reputation'):
            context_parts.append(f"Email Domain Reputation: {resume_info.get('domain_reputation')}")
        
        # Email verification detailed results
        email_verification = verification_result.get('email_verification', {})
        if email_verification:
            if email_verification.get('mx_records'):
                context_parts.append(f"Email MX Records Valid: {email_verification['mx_records']}")
            if email_verification.get('disposable'):
                context_parts.append(f"Disposable Email: {email_verification['disposable']}")
            if email_verification.get('role_based'):
                context_parts.append(f"Role-based Email: {email_verification['role_based']}")
        
        # Social profiles verification
        social_profiles = verification_result.get('social_profiles', {})
        if social_profiles:
            context_parts.append("Social Profiles Verification:")
            for platform, data in social_profiles.items():
                if isinstance(data, dict):
                    status = "Verified" if data.get('verified') else "Found but not verified"
                    context_parts.append(f"  {platform}: {status}")
                    if data.get('url'):
                        context_parts.append(f"    URL: {data['url']}")
        
        # Authenticity score calculation
        auth_score = resume_info.get('authenticity_score', 0)
        if not auth_score and verification_result.get('profiles'):
            # Calculate based on profiles found
            profiles_found = verification_result['profiles']
            score = 0
            if profiles_found.get('github', {}).get('found'):
                score += 50
            if profiles_found.get('linkedin', {}).get('found'):
                score += 50
            auth_score = score
        
        context_parts.append(f"Authenticity Score: {auth_score}/100")
        
        # Verification summary details
        if verification_result.get('verification_summary'):
            context_parts.append(f"Verification Summary: {verification_result['verification_summary']}")
        
        # Profiles summary
        profiles_summary = []
        if github_data and github_data.get('found'):
            profiles_summary.append("GitHub")
        if linkedin_data and linkedin_data.get('found'):
            profiles_summary.append("LinkedIn")
        if scraped_content:
            profiles_summary.append(f"{len(scraped_content)} other sites")
        
        if profiles_summary:
            context_parts.append(f"Online Profiles Found: {', '.join(profiles_summary)}")
    
    # Contact information and social links
    context_parts.append("\n=== CONTACT & SOCIAL INFORMATION ===")
    if resume_info.get('email'):
        context_parts.append(f"Primary Email: {resume_info['email']}")
    if resume_info.get('phone'):
        context_parts.append(f"Phone: {resume_info['phone']}")
    
    # Extract social links from profiles
    social_links = []
    if github_data and github_data.get('profile_url'):
        social_links.append(f"GitHub: {github_data['profile_url']}")
    if linkedin_data and linkedin_data.get('verified_urls'):
        urls = linkedin_data['verified_urls']
        if isinstance(urls, list) and urls:
            social_links.append(f"LinkedIn: {urls[0]}")
        elif isinstance(urls, str):
            social_links.append(f"LinkedIn: {urls}")
    
    if social_links:
        context_parts.append("Social Media Profiles:")
        for link in social_links:
            context_parts.append(f"  - {link}")
    
    # Additional verification insights
    if verification_result and not verification_result.get('error'):
        context_parts.append("\n=== VERIFICATION INSIGHTS ===")
        
        # Cross-verification between resume and online presence
        resume_skills = set(skill.lower() for skill in resume_info.get('skills', []))
        online_skills = set()
        
        # Collect skills from GitHub
        if github_data and github_data.get('languages'):
            online_skills.update(lang.lower() for lang in github_data['languages'])
        
        # Collect skills from scraped content
        for content in scraped_content.values():
            if content and isinstance(content, dict) and content.get('skills'):
                skills = content['skills']
                if isinstance(skills, list):
                    online_skills.update(skill.lower() for skill in skills)
                else:
                    online_skills.add(str(skills).lower())
        
        # Find skill overlaps and discrepancies
        skill_overlap = resume_skills.intersection(online_skills)
        resume_only_skills = resume_skills - online_skills
        online_only_skills = online_skills - resume_skills
        
        if skill_overlap:
            context_parts.append(f"Skills Verified Online: {', '.join(skill_overlap)}")
        if resume_only_skills:
            context_parts.append(f"Resume Skills Not Found Online: {', '.join(resume_only_skills)}")
        if online_only_skills:
            context_parts.append(f"Additional Skills Found Online: {', '.join(online_only_skills)}")
    
    # Join all context parts
    comprehensive_context = "\n".join(context_parts)
    
    # Create the prompt with comprehensive context
    prompt = f"""You are an expert HR assistant and candidate evaluator. You have access to comprehensive information about a candidate including their resume, online presence, and job match analysis.

CANDIDATE INFORMATION:
{comprehensive_context}

USER QUESTION: {user_input}

INSTRUCTIONS:
- Answer the question based on the comprehensive candidate information provided above
- Be specific and reference actual data from the candidate's profile
- If asked about skills, mention both resume skills and any found in their online presence
- If asked about experience, include both resume experience and LinkedIn/GitHub activity
- If asked about projects, reference both resume projects and GitHub repositories
- Provide insights based on the match analysis scores when relevant
- If information is not available, clearly state what is missing
- Be conversational but professional
- Keep responses focused and relevant to the question asked
- Use the verification data to assess candidate authenticity when relevant

Please provide a detailed, helpful response based on all available candidate information:"""
    
    try:
        response = call_groq_api(prompt)
        if isinstance(response, dict) and "error" in response:
            return f"I apologize, but I'm having trouble connecting to the chat service: {response['error']}"
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

# Main app UI
def main():
    st.title("Automated Resume Screening & Ranking System")
    
    # Reset chat state on page load
    if 'page_reloaded' not in st.session_state:
        st.session_state.page_reloaded = True
        # Clear all chat-related session state
        for key in list(st.session_state.keys()):
            if any(key.startswith(prefix) for prefix in ['current_', 'question_', 'saved_']):
                del st.session_state[key]
    
    # Initialize session state for shortlisted candidates
    if 'shortlisted_candidates' not in st.session_state:
        st.session_state.shortlisted_candidates = set()
    
    tab1, tab2, tab3 = st.tabs(["Resume Screening", "Results", "📊 Analytics & Insights"])

    with tab1:
        st.header("Upload Resumes & Job Description")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1️⃣ Upload Resumes")
            upload_option = st.radio("Choose upload method:", ["Individual PDFs", "Zip File (Multiple PDFs)"])
            
            resume_data = {}
            
            if upload_option == "Individual PDFs":
                uploaded_files = st.file_uploader("Upload resumes (PDF only)", type=["pdf"], accept_multiple_files=True)
                if uploaded_files:
                    with st.spinner("Processing uploaded PDFs..."):
                        for file in uploaded_files:
                            extracted = extract_text_from_pdf(file)
                            if extracted["text"]:
                                resume_data[file.name] = extracted
                        
                        st.success(f"✅ Successfully processed {len(resume_data)} resumes")
                        
                        # Show if any embedded links were found
                        total_links = sum(len(data["links"]) for data in resume_data.values())
                        if total_links > 0:
                            st.info(f"📎 Found {total_links} embedded links across all resumes")
            
            else:  # Zip File
                zip_file = st.file_uploader("Upload ZIP file containing resumes (PDFs only)", type=["zip"])
                if zip_file:
                    with st.spinner("Extracting resumes from ZIP file..."):
                        resume_data = handle_zip_upload(zip_file)
                        st.success(f"✅ Successfully extracted {len(resume_data)} resumes from ZIP file")
                        
                        # Show if any embedded links were found
                        total_links = sum(len(data["links"]) for data in resume_data.values())
                        if total_links > 0:
                            st.info(f"📎 Found {total_links} embedded links across all resumes")
            
            # Show the count and names of processed resumes
            if resume_data:
                with st.expander("View processed resume files"):
                    for i, (filename, data) in enumerate(resume_data.items(), 1):
                        link_count = len(data["links"])
                        if link_count > 0:
                            st.text(f"{i}. {filename} (📎 {link_count} links)")
                        else:
                            st.text(f"{i}. {filename}")
        
        with col2:
            st.subheader("2️⃣ Job Description & Requirements")
            
            job_description = st.text_area(
                "Enter the job description:", 
                height=250,
                placeholder="Paste the complete job description here...",
                key="job_desc_input"
            ).strip()
            
            # Enhanced filtering options
            st.subheader("Filtering Options")
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                min_experience = st.number_input("Minimum Years of Experience", min_value=0, value=0)
                required_skills = st.multiselect(
                    "Required Skills",
                    ["Python", "Java", "JavaScript", "React", "Node.js", "SQL", "AWS", "Docker"],
                    default=[]
                )
            
            with col_filter2:
                preferred_region = st.text_input(
                    "Region:",
                    placeholder="e.g., New York, Remote, etc."
                )
                education_level = st.multiselect(
                    "Education Level",
                    ["Bachelor's", "Master's", "PhD", "Other"],
                    default=[]
                )
            
            top_n = st.number_input(
                "Number of top candidates to show:",
                min_value=1,
                max_value=100,
                value=10
            )
        
        if st.button("🔍 Screen & Rank Resumes", type="primary", use_container_width=True, key="main_screening_button") and resume_data and job_description:
            if not GROQ_API_KEYS:
                st.error("❌ No Groq API keys available. Please add them to your .env file as GROQ_API_KEY_1, GROQ_API_KEY_2, etc.")
            else:
                # Store results in session state for tab2
                if "screening_results" not in st.session_state:
                    st.session_state.screening_results = {}
                
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Create a progress bar
                progress_bar = progress_placeholder.progress(0)
                
                # Step 1: Analyze job description
                status_placeholder.info("📝 Analyzing job requirements...")
                progress_bar.progress(0.2)
                
                job_requirements = analyze_job_description(job_description)
                
                if not job_requirements:
                    st.error("❌ Failed to analyze job description")
                    return
                
                # Step 2: Build vector database
                status_placeholder.info("🔄 Building candidate database...")
                progress_bar.progress(0.4)
                
                index, vectorizer, job_vector = build_faiss_index(resume_data, job_requirements)
                faiss_results = search_relevant_resumes(index, job_vector, resume_data)
                
                # Step 3: Process resumes
                status_placeholder.info("🔍 Screening candidates...")
                progress_bar.progress(0.6)
                
                total_resumes = len(faiss_results)
                detailed_results = []
                batch_size = min(5, max(1, total_resumes // 10))
                batches = [faiss_results[i:i + batch_size] for i in range(0, len(faiss_results), batch_size)]
                
                for batch_index, batch in enumerate(batches):
                    batch_results = process_resume_batch(batch, resume_data, job_requirements)
                    detailed_results.extend(batch_results)
                    progress_value = 0.6 + (0.4 * (batch_index + 1) / len(batches))
                    progress_bar.progress(progress_value)
                
                # Clear progress indicators
                progress_placeholder.empty()
                status_placeholder.empty()
                
                # Sort and store results
                sorted_results = sorted(
                    detailed_results,
                    key=lambda x: x["match_result"].get("scores", {}).get("overall_match", 0),
                    reverse=True
                )
                
                if preferred_region:
                    filtered_results = filter_by_region(sorted_results, preferred_region)
                    if not filtered_results:
                        st.warning(f"No candidates found in the region: {preferred_region}")
                        st.session_state.screening_results = {"results": sorted_results, "job_requirements": job_requirements, "top_n": top_n}
                    else:
                        st.success(f"Found {len(filtered_results)} candidates in {preferred_region}")
                        st.session_state.screening_results = {"results": filtered_results, "job_requirements": job_requirements, "top_n": top_n}
                else:
                    st.session_state.screening_results = {"results": sorted_results, "job_requirements": job_requirements, "top_n": top_n}
                
                st.success("Screening complete! View results in the 'Results' tab")
    
    with tab2:
        if "screening_results" in st.session_state and st.session_state.screening_results:
            results = st.session_state.screening_results["results"]
            job_requirements = st.session_state.screening_results["job_requirements"]
            top_n = st.session_state.screening_results["top_n"]
            
            # Show results summary
            st.markdown("### 📊 Results Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Total Candidates: {len(results)}")
            with col2:
                if len(results) > 0:
                    st.write(f"Top Score: {results[0]['match_result']['scores']['overall_match']:.1f}%")
                    st.write(f"Average Score: {sum(r['match_result']['scores']['overall_match'] for r in results) / len(results):.1f}%")
            with col3:
                if st.button("📊 View Analytics Dashboard", type="secondary", key="view_analytics"):
                    st.session_state.show_analytics = True
                if st.button("📈 Generate Insights Report", type="secondary", key="generate_insights"):
                    st.session_state.show_insights_report = True
            
            # Display results in card format
            st.markdown("### 👥 Candidate Results")
            
            # Initialize session state for selected candidates if not exists
            if 'selected_candidates' not in st.session_state:
                st.session_state.selected_candidates = set()
            
            if 'show_filters' not in st.session_state:
                st.session_state.show_filters = False
            
            if 'current_candidate' not in st.session_state:
                st.session_state.current_candidate = None
            
            # Bulk actions for selected candidates
            if st.session_state.selected_candidates:
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Selected: {len(st.session_state.selected_candidates)} candidates**")
                with col2:
                    if st.button("📧 Send Email to Selected", type="primary", key="bulk_email_main"):
                        st.session_state.show_bulk_email = True
                with col3:
                    if st.button("📋 Export Selected", key="export_main"):
                        # Create export data for selected candidates
                        selected_data = []
                        for result in results:
                            candidate_name = result["resume_info"].get("name", "Unknown")
                            if candidate_name in st.session_state.selected_candidates:
                                selected_data.append({
                                    "Name": candidate_name,
                                    "Email": result["resume_info"].get("email", ""),
                                    "Overall Match": f"{result['match_result']['scores']['overall_match']:.1f}%",
                                    "Technical Score": f"{result['match_result']['scores']['technical_skills']:.1f}%",
                                    "Experience Score": f"{result['match_result']['scores']['experience']:.1f}%"
                                })
                        
                        if selected_data:
                            df_selected = pd.DataFrame(selected_data)
                            st.markdown(get_csv_download_link(df_selected, "selected_candidates.csv"), unsafe_allow_html=True)
                st.markdown("---")
            
            # Create three columns for cards with selection
            for i in range(0, len(results), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(results):
                        with cols[j]:
                            result = results[i + j]
                            resume_info = result.get("resume_info", {})
                            match_result = result.get("match_result", {})
                            scores = match_result.get("scores", {})
                            
                            # Create candidate card using Streamlit components
                            candidate_name = resume_info.get('name', 'Unknown')
                            candidate_email = resume_info.get('email', '')
                            
                            # Create a container for the card
                            with st.container():
                                # Checkbox for selection
                                is_selected = st.checkbox(
                                    f"Select {candidate_name}",
                                    value=candidate_name in st.session_state.selected_candidates,
                                    key=f"checkbox_{i+j}"
                                )
                                
                                # Update selection state
                                if is_selected:
                                    st.session_state.selected_candidates.add(candidate_name)
                                else:
                                    st.session_state.selected_candidates.discard(candidate_name)
                                
                                # Visual indicator for selected candidates
                                if is_selected:
                                    st.markdown("✅ **SELECTED**")
            
                                # Candidate name and match score
                                st.markdown(f"### {candidate_name}")
                                st.markdown(f"**Match Score: {scores.get('overall_match', 0):.1f}%**")
                                
                                # Progress bar for match score
                                st.progress(scores.get('overall_match', 0) / 100)
                                
                                # Top skills
                                skills = resume_info.get("skills", [])[:3]
                                if skills:
                                    st.markdown("**Top Skills:**")
                                    skill_text = " • ".join(skills)
                                    st.markdown(f"*{skill_text}*")
                                
                                # Contact info with clickable email
                                if candidate_email:
                                    # Create mailto link with proper HTML
                                    mailto_link = f"mailto:{candidate_email}?subject=Job Opportunity - {candidate_name}&body=Dear {candidate_name},%0D%0A%0D%0AWe are interested in discussing a job opportunity with you.%0D%0A%0D%0ABest regards"
                                    st.markdown(f"<a href='{mailto_link}' target='_blank' style='text-decoration: none; color: inherit;'>📧 {candidate_email}</a>", unsafe_allow_html=True)
                                
                                # Social links
                                links = []
                                github_profile = result.get("profiles", {}).get("github", {})
                                if github_profile.get("profile_url"):
                                    links.append(f"[🐱 GitHub]({github_profile['profile_url']})")
                                
                                linkedin_profile = result.get("profiles", {}).get("linkedin", {})
                                if linkedin_profile.get("verified_urls"):
                                    links.append(f"[💼 LinkedIn]({linkedin_profile['verified_urls'][0]})")
                                
                                if links:
                                    st.markdown(" | ".join(links))
                    
                                # View details button
                                if st.button(
                                    "👁️ View Details", 
                                    key=f"details_btn_{i+j}",
                                    use_container_width=True
                                ):
                                    st.session_state.current_candidate = candidate_name
                                    st.session_state.show_details = True
                                    st.rerun()
                                
                                st.markdown("---")
            
            # Show detailed view for current candidate
            if st.session_state.current_candidate:
                st.markdown("---")
                st.markdown(f"### Candidate Details: {st.session_state.current_candidate}")
                    
                # Find the selected candidate's data
                selected_result = next(
                    (r for r in results if r["resume_info"].get("name") == st.session_state.current_candidate),
                    None
                )
                
                if selected_result:
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("💬 Start Q&A", key="start_qa"):
                            st.session_state.show_chat = True
                    
                    with col2:
                        if st.button("🌐 Web Presence", key="web_presence"):
                            st.session_state.show_web_data = True
                    
                    with col3:
                        if st.button("📧 Send Email", key="send_email_single"):
                            candidate_email = selected_result["resume_info"].get("email", "")
                            candidate_name = selected_result["resume_info"].get("name", "Unknown")
                            if candidate_email:
                                mailto_link = f"mailto:{candidate_email}?subject=Job Opportunity - {candidate_name}&body=Dear {candidate_name},%0D%0A%0D%0AWe are interested in discussing a job opportunity with you.%0D%0A%0D%0ABest regards"
                                st.markdown(f"<a href='{mailto_link}' target='_blank' style='text-decoration: none; color: inherit;'>📧 Open Email Client</a>", unsafe_allow_html=True)
                            else:
                                st.error("No email address found for this candidate")
                    
                    # Always show details for current candidate
                    with st.expander("📋 Candidate Details", expanded=True):
                        resume_info = selected_result["resume_info"]
                        match_result = selected_result["match_result"]
                        scores = match_result.get("scores", {})
                        
                        col1, col2 = st.columns([2,1])
                        
                        with col1:
                            st.markdown("#### Contact Information")
                            if resume_info.get("email"):
                                candidate_email = resume_info["email"]
                                candidate_name = resume_info.get("name", "Unknown")
                                if st.button(f"📧 {candidate_email}", key=f"email_btn_{candidate_name}"):
                                    mailto_link = f"mailto:{candidate_email}?subject=Job Opportunity - {candidate_name}&body=Dear {candidate_name},%0D%0A%0D%0AWe are interested in discussing a job opportunity with you.%0D%0A%0D%0ABest regards"
                                    st.markdown(f"<a href='{mailto_link}' target='_blank' style='display: none;' id='email_link'></a>", unsafe_allow_html=True)
                                    st.markdown("""
                                        <script>
                                            document.getElementById('email_link').click();
                                        </script>
                                    """, unsafe_allow_html=True)
                            
                            if resume_info.get("phone"):
                                st.markdown(f"📞 {resume_info['phone']}")
                            
                            # Social links
                            social_links = []
                            github_profile = selected_result.get("profiles", {}).get("github", {})
                            if github_profile.get("profile_url"):
                                social_links.append(f"🐱 [GitHub]({github_profile['profile_url']})")
                            
                            linkedin_profile = selected_result.get("profiles", {}).get("linkedin", {})
                            if linkedin_profile.get("verified_urls"):
                                social_links.append(f"💼 [LinkedIn]({linkedin_profile['verified_urls'][0]})")
                            
                            if social_links:
                                st.markdown(" | ".join(social_links), unsafe_allow_html=True)
                    
                        with col2:
                            st.markdown("#### Match Scores")
                            st.progress(scores.get("overall_match", 0) / 100)
                            st.write(f"Overall: {scores.get('overall_match', 0):.1f}%")
                            st.write(f"Technical: {scores.get('technical_skills', 0):.1f}%")
                            st.write(f"Experience: {scores.get('experience', 0):.1f}%")
                        
                        st.markdown("#### Skills")
                        skills = resume_info.get("skills", [])
                        if skills:
                            skill_cols = st.columns(3)
                            for idx, skill in enumerate(skills):
                                with skill_cols[idx % 3]:
                                    st.markdown(f"• {skill}")
                        else:
                            st.info("No skills listed in resume")
                        
                        st.markdown("#### Experience")
                        experience = resume_info.get("experience", [])
                        if experience:
                            for exp in experience:
                                st.markdown(f"• {exp}")
                        else:
                            st.info("No experience details found")
                        
                        st.markdown("#### Education")
                        education = resume_info.get("education", [])
                        if education:
                            for edu in education:
                                st.markdown(f"• {edu}")
                        else:
                            st.info("No education details found")
                    
                    # Show selected content based on button clicks
                    if hasattr(st.session_state, 'show_chat') and st.session_state.show_chat:
                        with st.expander("💬 Pre-screening Q&A", expanded=True):
                            # Check Groq API connection
                            st.info("💡 Using Groq API for chat responses")
                            
                            if "chat_history" not in st.session_state:
                                st.session_state.chat_history = {}
                            
                            candidate_id = f"chat_{st.session_state.current_candidate}"
                            if candidate_id not in st.session_state.chat_history:
                                st.session_state.chat_history[candidate_id] = []
                            
                            # Show chat history
                            for msg in st.session_state.chat_history[candidate_id]:
                                if msg["role"] == "user":
                                    st.markdown(f"🤔 **You:** {msg['content']}")
                                else:
                                    st.markdown(f"🤖 **AI:** {msg['content']}")
                            
                            # Chat input with form to prevent auto-rerun
                            with st.form(key=f"chat_form_{candidate_id}", clear_on_submit=True):
                                user_input = st.text_input(
                                    "Ask about this candidate:",
                                    placeholder="e.g., What are their main technical skills?",
                                    key=f"chat_input_{candidate_id}"
                                )
                                submit_button = st.form_submit_button("Send")
                                
                                if submit_button and user_input.strip():
                                    with st.spinner("🤖 Generating response..."):
                                        # Prepare context for AI
                                        context = {
                                            "resume_info": selected_result["resume_info"],
                                            "match_result": selected_result["match_result"],
                                            "scraped_content": selected_result.get("scraped_content", {}),
                                            "profiles": selected_result.get("profiles", {}),
                                            "verification_result": selected_result.get("verification_result", {})
                                        }
                                        
                                        # Generate AI response
                                        try:
                                            response = generate_chat_response(user_input, context)
                                            
                                            # Add to chat history
                                            st.session_state.chat_history[candidate_id].append(
                                                {"role": "user", "content": user_input}
                                            )
                                            st.session_state.chat_history[candidate_id].append(
                                                {"role": "assistant", "content": response}
                                            )
                                            
                                            # Rerun to show new messages
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"Chat error: {str(e)}")
                                            st.info("Please ensure your Groq API key is properly configured")
                    
                    if hasattr(st.session_state, 'show_web_data') and st.session_state.show_web_data:
                        with st.expander("🌐 Web Presence", expanded=True):
                            verification_result = selected_result.get("verification_result", {})
                            
                            # Check if we have any data to display
                            has_data = False
                            
                            # GitHub information
                            github_data = selected_result.get("profiles", {}).get("github", {})
                            if github_data and github_data.get("found"):
                                has_data = True
                                st.markdown("#### 🐱 GitHub Profile")
                                
                                if github_data.get("profile_url"):
                                    st.markdown(f"**Profile:** [{github_data['profile_url']}]({github_data['profile_url']})")
                                
                                if github_data.get("bio"):
                                    st.markdown(f"**Bio:** {github_data['bio']}")
                        
                                # Display public repositories count
                                public_repos = github_data.get("public_repos", 0)
                                if public_repos > 0:
                                    st.markdown(f"**Public Repositories:** {public_repos}")
                                
                                if github_data.get("company"):
                                    st.markdown(f"**Company:** {github_data['company']}")
                                
                                if github_data.get("location"):
                                    st.markdown(f"**Location:** {github_data['location']}")
                                
                                if github_data.get("blog"):
                                    st.markdown(f"**Website/Blog:** {github_data['blog']}")
                                
                                # Display repositories in a more prominent way
                                if github_data.get("repositories"):
                                    st.markdown("\n### 📚 Repositories")
                                    repos = github_data.get("repositories", [])
                                    if isinstance(repos, list) and repos:
                                        # Ensure at least one repository is displayed
                                        for repo in repos[:1]:  # Show only the first repository
                                            if isinstance(repo, dict):
                                                with st.container():
                                                    name = repo.get('name', 'Unknown')
                                                    url = repo.get('url', '#')
                                                    st.markdown(f"**[{name}]({url})**")
                                            else:
                                                st.markdown(f"• {repo}")
                                
                                st.markdown("---")
                            
                            # LinkedIn information
                            linkedin_data = selected_result.get("profiles", {}).get("linkedin", {})
                            if linkedin_data and linkedin_data.get("found"):
                                has_data = True
                                st.markdown("#### 💼 LinkedIn Profile")
                                
                                if linkedin_data.get("verified_urls"):
                                    urls = linkedin_data["verified_urls"]
                                    if isinstance(urls, list) and urls:
                                        st.markdown(f"**Profile:** [{urls[0]}]({urls[0]})")
                                        linkedin_url = urls[0]
                                        username = linkedin_url.split('/')[-1]  # Extract username from URL
                                        
                                        # LinkedIn recent activity API call
                                        try:
                                            activity_url = "https://linkedin-data-api.p.rapidapi.com/profiles/recent-activity/all"
                                            activity_payload = {
                                                "username": username,
                                                "page": 1,
                                                "limit": 1  # Get only 1 most recent activity
                                            }
                                            activity_headers = {
                                                "x-rapidapi-key": "512175d384mshb7894ebd80ddda4p11a2c4jsn7fbc012751c2",
                                                "x-rapidapi-host": "linkedin-data-api.p.rapidapi.com",
                                                "Content-Type": "application/json"
                                            }
                                            
                                            activity_response = requests.post(
                                                activity_url,
                                                json=activity_payload,
                                                headers=activity_headers
                                            )
                                            
                                            if activity_response.status_code == 200:
                                                activity_data = activity_response.json()
                                                if activity_data and isinstance(activity_data, dict):
                                                    st.markdown("\n### Recent Activity")
                                                    if 'activities' in activity_data and activity_data['activities']:
                                                        activity = activity_data['activities'][0]  # Get the first activity
                                                        with st.container():
                                                            activity_type = activity.get('type', 'Unknown')
                                                            activity_date = activity.get('date', 'Unknown date')
                                                            activity_text = activity.get('text', 'No text available')
                                                            activity_likes = activity.get('likes', 0)
                                                            activity_comments = activity.get('comments', 0)
                                                            
                                                            # Display activity type with appropriate emoji
                                                            activity_emoji = {
                                                                'post': '📝',
                                                                'article': '📰',
                                                                'comment': '💬',
                                                                'like': '👍',
                                                                'share': '🔄',
                                                                'job': '💼',
                                                                'connection': '🤝',
                                                                'certification': '🏆',
                                                                'default': '📌'
                                                            }.get(activity_type.lower(), '📌')
                                                            
                                                            st.markdown(f"{activity_emoji} **{activity_type.title()}** on {activity_date}")
                                                            st.markdown(f"{activity_text[:200]}..." if len(activity_text) > 200 else activity_text)
                                                            
                                                            # Show engagement metrics if available
                                                            if activity_likes > 0 or activity_comments > 0:
                                                                st.markdown(f"👍 {activity_likes} likes | 💬 {activity_comments} comments")
                                                            
                                                            # Show additional activity details if available
                                                            if activity.get('company'):
                                                                st.markdown(f"🏢 Company: {activity['company']}")
                                                            if activity.get('title'):
                                                                st.markdown(f"📋 Title: {activity['title']}")
                                                            if activity.get('url'):
                                                                st.markdown(f"[View Activity]({activity['url']})")
                                        except Exception as e:
                                            st.markdown(f"Note: Could not fetch recent activity: {str(e)}")
                                    elif isinstance(urls, str):
                                        st.markdown(f"**Profile:** [{urls}]({urls})")
                                
                                st.markdown("---")
            
            # Bulk email functionality
            if hasattr(st.session_state, 'show_bulk_email') and st.session_state.show_bulk_email:
                st.markdown("---")
                with st.expander("📧 Send Bulk Email", expanded=True):
                    # Get selected candidates' emails
                    selected_emails = []
                    selected_names = []
                    for result in results:
                        candidate_name = result["resume_info"].get("name", "Unknown")
                        if candidate_name in st.session_state.selected_candidates:
                            email = result["resume_info"].get("email", "")
                            if email:
                                selected_emails.append(email)
                                selected_names.append(candidate_name)
                    
                    if selected_emails:
                        st.write(f"**Selected Recipients ({len(selected_emails)}):**")
                        for name, email in zip(selected_names, selected_emails):
                            st.write(f"• {name} ({email})")
                        
                        # Email composition
                        email_subject = st.text_input("Subject:", value="Job Opportunity")
                        email_body = st.text_area("Message:", value="""Dear Candidate,

We are interested in discussing a job opportunity with you based on your profile.

Please let us know if you're interested in learning more about this position.

Best regards,
HR Team""", height=200)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📧 Open Email Client (All Recipients)", type="primary", key="bulk_email_all"):
                                # Create mailto link with all recipients
                                recipients = ";".join(selected_emails)
                                mailto_link = f"mailto:{recipients}?subject={email_subject}&body={email_body.replace(chr(10), '%0D%0A')}"
                                st.markdown(f"<a href='{mailto_link}' target='_blank'>📧 Open Email Client with All Recipients</a>", unsafe_allow_html=True)
                        
                        with col2:
                            if st.button("📧 Individual Email Links", key="bulk_email_individual"):
                                st.write("**Individual email links:**")
                                for name, email in zip(selected_names, selected_emails):
                                    personal_body = email_body.replace("Dear Candidate", f"Dear {name}")
                                    mailto_link = f"mailto:{email}?subject={email_subject}&body={personal_body.replace(chr(10), '%0D%0A')}"
                                    st.markdown(f"<a href='{mailto_link}' target='_blank'>📧 {name}</a>", unsafe_allow_html=True)
                        
                        if st.button("❌ Close Email Composer", key="close_bulk_email"):
                            del st.session_state.show_bulk_email
                            st.rerun()
                    else:
                        st.warning("No email addresses found for selected candidates")
                        if st.button("❌ Close", key="close_bulk_email_warning"):
                            del st.session_state.show_bulk_email
                            st.rerun()
            
            # Show shortlisted candidates
            if st.session_state.selected_candidates:
                st.sidebar.markdown("### ⭐ Selected Candidates")
                for candidate in st.session_state.selected_candidates:
                    st.sidebar.markdown(f"• {candidate}")
                
                if st.sidebar.button("📧 Send Email to All Selected", key="bulk_email_sidebar"):
                    st.session_state.show_bulk_email = True
                    st.rerun()
                
                if st.sidebar.button("📋 Export Selected", key="export_sidebar"):
                    selected_data = []
                    for result in results:
                        candidate_name = result["resume_info"].get("name", "Unknown")
                        if candidate_name in st.session_state.selected_candidates:
                            selected_data.append({
                                "Name": candidate_name,
                                "Email": result["resume_info"].get("email", ""),
                                "Overall Match": f"{result['match_result']['scores']['overall_match']:.1f}%",
                                "Technical Score": f"{result['match_result']['scores']['technical_skills']:.1f}%",
                                "Experience Score": f"{result['match_result']['scores']['experience']:.1f}%"
                            })
                    
                    if selected_data:
                        df_selected = pd.DataFrame(selected_data)
                        st.sidebar.markdown(get_csv_download_link(df_selected, "selected_candidates.csv"), unsafe_allow_html=True)
            
            # Analytics Dashboard
            if hasattr(st.session_state, 'show_analytics') and st.session_state.show_analytics:
                st.markdown("---")
                with st.expander("📊 Talent Pool Analytics Dashboard", expanded=True):
                    # Calculate analytics metrics
                    total_candidates = len(results)
                    avg_score = sum(r['match_result']['scores']['overall_match'] for r in results) / total_candidates if total_candidates > 0 else 0
                    
                    # Score distribution
                    scores = [r['match_result']['scores']['overall_match'] for r in results]
                    score_ranges = {
                        "Excellent (90-100%)": len([s for s in scores if s >= 90]),
                        "Good (80-89%)": len([s for s in scores if 80 <= s < 90]),
                        "Average (70-79%)": len([s for s in scores if 70 <= s < 80]),
                        "Below Average (60-69%)": len([s for s in scores if 60 <= s < 70]),
                        "Poor (<60%)": len([s for s in scores if s < 60])
                    }
                    
                    # Skills analysis
                    all_skills = []
                    for result in results:
                        skills = result["resume_info"].get("skills", [])
                        all_skills.extend([skill.lower() for skill in skills])
                    
                    skill_counts = {}
                    for skill in all_skills:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
                    
                    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Candidates", total_candidates)
                        st.metric("Average Match Score", f"{avg_score:.1f}%")
                    
                    with col2:
                        excellent_count = score_ranges["Excellent (90-100%)"]
                        good_count = score_ranges["Good (80-89%)"]
                        st.metric("Excellent Matches", excellent_count, f"{(excellent_count/total_candidates*100):.1f}%" if total_candidates > 0 else "0%")
                        st.metric("Good Matches", good_count, f"{(good_count/total_candidates*100):.1f}%" if total_candidates > 0 else "0%")
                    
                    with col3:
                        verified_candidates = sum(1 for r in results if r["resume_info"].get("verified_email", False))
                        github_profiles = sum(1 for r in results if r.get("profiles", {}).get("github", {}).get("found", False))
                        st.metric("Verified Email", verified_candidates, f"{(verified_candidates/total_candidates*100):.1f}%" if total_candidates > 0 else "0%")
                        st.metric("GitHub Profiles", github_profiles, f"{(github_profiles/total_candidates*100):.1f}%" if total_candidates > 0 else "0%")
                    
                    with col4:
                        linkedin_profiles = sum(1 for r in results if r.get("profiles", {}).get("linkedin", {}).get("found", False))
                        avg_tech_score = sum(r['match_result']['scores']['technical_skills'] for r in results) / total_candidates if total_candidates > 0 else 0
                        st.metric("LinkedIn Profiles", linkedin_profiles, f"{(linkedin_profiles/total_candidates*100):.1f}%" if total_candidates > 0 else "0%")
                        st.metric("Avg Technical Score", f"{avg_tech_score:.1f}%")
                    
                    st.markdown("---")
                    
                    # Score Distribution Chart
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 Score Distribution")
                        score_df = pd.DataFrame(list(score_ranges.items()), columns=['Range', 'Count'])
                        st.bar_chart(score_df.set_index('Range'))
                    
                    with col2:
                        st.markdown("#### 🏆 Top Skills in Talent Pool")
                        if top_skills:
                            for i, (skill, count) in enumerate(top_skills[:10], 1):
                                percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
                                st.write(f"{i}. **{skill.title()}** - {count} candidates ({percentage:.1f}%)")
                        else:
                            st.info("No skills data available")
                    
                    # Experience Analysis
                    st.markdown("---")
                    st.markdown("#### 💼 Experience Analysis")
                    
                    experience_scores = [r['match_result']['scores']['experience'] for r in results]
                    education_scores = [r['match_result']['scores']['education'] for r in results]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_exp_score = sum(experience_scores) / len(experience_scores) if experience_scores else 0
                        st.metric("Average Experience Score", f"{avg_exp_score:.1f}%")
                        exp_excellent = len([s for s in experience_scores if s >= 90])
                        st.write(f"Excellent Experience: {exp_excellent} candidates")
                    
                    with col2:
                        avg_edu_score = sum(education_scores) / len(education_scores) if education_scores else 0
                        st.metric("Average Education Score", f"{avg_edu_score:.1f}%")
                        edu_excellent = len([s for s in education_scores if s >= 90])
                        st.write(f"Excellent Education: {edu_excellent} candidates")
                    
                    with col3:
                        # Certification analysis
                        total_certs = sum(len(r["resume_info"].get("certifications", [])) for r in results)
                        candidates_with_certs = sum(1 for r in results if len(r["resume_info"].get("certifications", [])) > 0)
                        st.metric("Total Certifications", total_certs)
                        st.write(f"Candidates with Certs: {candidates_with_certs}")
                    
                    if st.button("❌ Close Analytics Dashboard", key="close_analytics"):
                        del st.session_state.show_analytics
                        st.rerun()
            
            # Insights Report
            if hasattr(st.session_state, 'show_insights_report') and st.session_state.show_insights_report:
                st.markdown("---")
                with st.expander("📈 AI-Powered Insights Report", expanded=True):
                    with st.spinner("🤖 Generating insights using AI..."):
                        # Prepare data for AI analysis
                        analytics_data = {
                            "total_candidates": len(results),
                            "average_score": sum(r['match_result']['scores']['overall_match'] for r in results) / len(results) if results else 0,
                            "score_distribution": {
                                "excellent": len([r for r in results if r['match_result']['scores']['overall_match'] >= 90]),
                                "good": len([r for r in results if 80 <= r['match_result']['scores']['overall_match'] < 90]),
                                "average": len([r for r in results if 70 <= r['match_result']['scores']['overall_match'] < 80]),
                                "below_average": len([r for r in results if r['match_result']['scores']['overall_match'] < 70])
                            },
                            "top_performers": [
                                {
                                    "name": r["resume_info"].get("name", "Unknown"),
                                    "score": r['match_result']['scores']['overall_match'],
                                    "skills": r["resume_info"].get("skills", [])[:5]
                                } for r in results[:5]
                            ],
                            "skills_analysis": {},
                            "verification_stats": {
                                "verified_emails": sum(1 for r in results if r["resume_info"].get("verified_email", False)),
                                "github_profiles": sum(1 for r in results if r.get("profiles", {}).get("github", {}).get("found", False)),
                                "linkedin_profiles": sum(1 for r in results if r.get("profiles", {}).get("linkedin", {}).get("found", False))
                            }
                        }
                        
                        # Generate AI insights
                        insights_prompt = f"""
                        As a senior HR analytics expert, analyze this talent pool data and provide actionable insights:
                        
                        TALENT POOL DATA:
                        - Total Candidates: {analytics_data['total_candidates']}
                        - Average Match Score: {analytics_data['average_score']:.1f}%
                        - Score Distribution: {analytics_data['score_distribution']}
                        - Verification Stats: {analytics_data['verification_stats']}
                        
                        TOP PERFORMERS:
                        {chr(10).join([f"• {p['name']}: {p['score']:.1f}% - Skills: {', '.join(p['skills'])}" for p in analytics_data['top_performers']])}
                        
                        Provide insights on:
                        1. Overall talent pool quality
                        2. Hiring recommendations
                        3. Skills gaps and strengths
                        4. Market competitiveness
                        5. Diversity and verification insights
                        6. Action items for recruiters
                        
                        Format as a professional report with clear sections and actionable recommendations.
                        """
                        
                        try:
                            insights_response = call_groq_api(insights_prompt, model="llama3-70b-8192")
                            
                            if isinstance(insights_response, dict) and "error" in insights_response:
                                st.error(f"Error generating insights: {insights_response['error']}")
                                st.info("AI insights temporarily unavailable. Please check your API configuration.")
                            else:
                                st.markdown("### 🎯 AI-Generated Talent Pool Insights")
                                st.markdown(insights_response)
                                
                                # Export options
                                st.markdown("---")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Create downloadable report
                                    report_data = {
                                        "report_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "analytics_summary": analytics_data,
                                        "ai_insights": insights_response
                                    }
                                    
                                    report_json = json.dumps(report_data, indent=2)
                                    b64_report = base64.b64encode(report_json.encode()).decode()
                                    download_link = f'<a href="data:application/json;base64,{b64_report}" download="talent_pool_insights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json" class="btn" style="background-color:#FF6B6B;color:white;padding:8px 12px;text-decoration:none;border-radius:4px;">📥 Download Full Report (JSON)</a>'
                                    st.markdown(download_link, unsafe_allow_html=True)
                                
                                with col2:
                                    if st.button("📧 Email Report to Team", key="email_insights"):
                                        # Create email with insights
                                        email_subject = f"Talent Pool Analytics Report - {datetime.now().strftime('%Y-%m-%d')}"
                                        email_body = f"""Talent Pool Analytics Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY METRICS:
- Total Candidates: {analytics_data['total_candidates']}
- Average Match Score: {analytics_data['average_score']:.1f}%
- Excellent Matches: {analytics_data['score_distribution']['excellent']}
- Verified Profiles: {analytics_data['verification_stats']['verified_emails']} emails, {analytics_data['verification_stats']['github_profiles']} GitHub, {analytics_data['verification_stats']['linkedin_profiles']} LinkedIn

AI INSIGHTS:
{insights_response[:500]}...

Full report attached.

Best regards,
HR Analytics Team"""
                                        
                                        mailto_link = f"mailto:?subject={email_subject}&body={email_body.replace(chr(10), '%0D%0A')}"
                                        st.markdown(f"[📧 Open Email Client]({mailto_link})")
                        
                        except Exception as e:
                            st.error(f"Error generating insights: {str(e)}")
                            st.info("Please ensure your API configuration is correct.")
                    
                    if st.button("❌ Close Insights Report", key="close_insights"):
                        del st.session_state.show_insights_report
                        st.rerun()
        else:
            st.info("Please complete the resume screening process in the 'Resume Screening' tab")

    with tab3:
        st.header("📊 Talent Pool Analytics & Insights Dashboard")
        
        if "screening_results" in st.session_state and st.session_state.screening_results:
            results = st.session_state.screening_results["results"]
            
            # Overview metrics
            st.markdown("### 🎯 Key Performance Indicators")
            
            total_candidates = len(results)
            if total_candidates > 0:
                avg_score = sum(r['match_result']['scores']['overall_match'] for r in results) / total_candidates
                
                # Top row metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Candidates", total_candidates)
                
                with col2:
                    st.metric("Average Match", f"{avg_score:.1f}%")
                
                with col3:
                    excellent_matches = len([r for r in results if r['match_result']['scores']['overall_match'] >= 90])
                    st.metric("Excellent Matches", excellent_matches, f"{(excellent_matches/total_candidates*100):.1f}%")
                
                with col4:
                    verified_emails = sum(1 for r in results if r["resume_info"].get("verified_email", False))
                    st.metric("Verified Emails", verified_emails, f"{(verified_emails/total_candidates*100):.1f}%")
                
                with col5:
                    avg_tech_score = sum(r['match_result']['scores']['technical_skills'] for r in results) / total_candidates
                    st.metric("Avg Tech Score", f"{avg_tech_score:.1f}%")
                
                st.markdown("---")
                
                # Detailed Analytics Section
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score Distribution
                    st.markdown("### 📈 Score Distribution Analysis")
                    
                    scores = [r['match_result']['scores']['overall_match'] for r in results]
                    score_ranges = {
                        "Excellent (90-100%)": len([s for s in scores if s >= 90]),
                        "Good (80-89%)": len([s for s in scores if 80 <= s < 90]),
                        "Average (70-79%)": len([s for s in scores if 70 <= s < 80]),
                        "Below Average (60-69%)": len([s for s in scores if 60 <= s < 70]),
                        "Poor (<60%)": len([s for s in scores if s < 60])
                    }
                    
                    # Create DataFrame for chart
                    score_df = pd.DataFrame(list(score_ranges.items()), columns=['Score Range', 'Count'])
                    st.bar_chart(score_df.set_index('Score Range'))
                    
                    # Score breakdown
                    for range_name, count in score_ranges.items():
                        percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
                        st.write(f"**{range_name}**: {count} candidates ({percentage:.1f}%)")
                
                with col2:
                    # Skills Analysis
                    st.markdown("### 🛠️ Skills Landscape")
                    
                    all_skills = []
                    for result in results:
                        skills = result["resume_info"].get("skills", [])
                        all_skills.extend([skill.lower().strip() for skill in skills])
                    
                    skill_counts = {}
                    for skill in all_skills:
                        if skill:  # Avoid empty skills
                            skill_counts[skill] = skill_counts.get(skill, 0) + 1
                    
                    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
                    
                    if top_skills:
                        # Create skills chart
                        skills_df = pd.DataFrame(top_skills[:10], columns=['Skill', 'Count'])
                        st.bar_chart(skills_df.set_index('Skill'))
                        
                        st.markdown("**Top 15 Skills in Pool:**")
                        for i, (skill, count) in enumerate(top_skills, 1):
                            percentage = (count / total_candidates * 100) if total_candidates > 0 else 0
                            st.write(f"{i}. **{skill.title()}** - {count} candidates ({percentage:.1f}%)")
                    else:
                        st.info("No skills data available for analysis")
                
                # Experience & Education Analysis
                st.markdown("---")
                st.markdown("### 🎓 Experience & Education Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 💼 Experience Scores")
                    exp_scores = [r['match_result']['scores']['experience'] for r in results]
                    avg_exp = sum(exp_scores) / len(exp_scores) if exp_scores else 0
                    
                    exp_ranges = {
                        "Excellent (90-100%)": len([s for s in exp_scores if s >= 90]),
                        "Good (80-89%)": len([s for s in exp_scores if 80 <= s < 90]),
                        "Average (70-79%)": len([s for s in exp_scores if 70 <= s < 80]),
                        "Needs Development (<70%)": len([s for s in exp_scores if s < 70])
                    }
                    
                    st.metric("Average Experience Score", f"{avg_exp:.1f}%")
                    for range_name, count in exp_ranges.items():
                        st.write(f"**{range_name}**: {count}")
                
                with col2:
                    st.markdown("#### 🎓 Education Scores")
                    edu_scores = [r['match_result']['scores']['education'] for r in results]
                    avg_edu = sum(edu_scores) / len(edu_scores) if edu_scores else 0
                    
                    edu_ranges = {
                        "Excellent (90-100%)": len([s for s in edu_scores if s >= 90]),
                        "Good (80-89%)": len([s for s in edu_scores if 80 <= s < 90]),
                        "Average (70-79%)": len([s for s in edu_scores if 70 <= s < 80]),
                        "Needs Enhancement (<70%)": len([s for s in edu_scores if s < 70])
                    }
                    
                    st.metric("Average Education Score", f"{avg_edu:.1f}%")
                    for range_name, count in edu_ranges.items():
                        st.write(f"**{range_name}**: {count}")
                
                with col3:
                    st.markdown("#### 🏆 Certifications")
                    total_certs = sum(len(r["resume_info"].get("certifications", [])) for r in results)
                    candidates_with_certs = sum(1 for r in results if len(r["resume_info"].get("certifications", [])) > 0)
                    avg_certs_per_candidate = total_certs / total_candidates if total_candidates > 0 else 0
                    
                    st.metric("Total Certifications", total_certs)
                    st.metric("Candidates with Certs", candidates_with_certs)
                    st.metric("Avg Certs/Candidate", f"{avg_certs_per_candidate:.1f}")
                
                # Verification & Online Presence
                st.markdown("---")
                st.markdown("### 🔍 Verification & Online Presence")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    github_profiles = sum(1 for r in results if r.get("profiles", {}).get("github", {}).get("found", False))
                    st.metric("GitHub Profiles", github_profiles, f"{(github_profiles/total_candidates*100):.1f}%")
                
                with col2:
                    linkedin_profiles = sum(1 for r in results if r.get("profiles", {}).get("linkedin", {}).get("found", False))
                    st.metric("LinkedIn Profiles", linkedin_profiles, f"{(linkedin_profiles/total_candidates*100):.1f}%")
                
                with col3:
                    both_profiles = sum(1 for r in results if 
                                      r.get("profiles", {}).get("github", {}).get("found", False) and 
                                      r.get("profiles", {}).get("linkedin", {}).get("found", False))
                    st.metric("Both Profiles", both_profiles, f"{(both_profiles/total_candidates*100):.1f}%")
                
                with col4:
                    avg_auth_score = sum(r["resume_info"].get("authenticity_score", 0) for r in results) / total_candidates if total_candidates > 0 else 0
                    st.metric("Avg Authenticity", f"{avg_auth_score:.1f}/100")
                
                # Top Performers Section
                st.markdown("---")
                st.markdown("### 🏆 Top Performers Analysis")
                
                top_performers = sorted(results, key=lambda x: x['match_result']['scores']['overall_match'], reverse=True)[:10]
                
                st.markdown("#### Top 10 Candidates Overview")
                performers_data = []
                for i, performer in enumerate(top_performers, 1):
                    performers_data.append({
                        "Rank": i,
                        "Name": performer["resume_info"].get("name", "Unknown"),
                        "Overall Score": f"{performer['match_result']['scores']['overall_match']:.1f}%",
                        "Technical": f"{performer['match_result']['scores']['technical_skills']:.1f}%",
                        "Experience": f"{performer['match_result']['scores']['experience']:.1f}%",
                        "Email Verified": "✅" if performer["resume_info"].get("verified_email", False) else "❌",
                        "GitHub": "✅" if performer.get("profiles", {}).get("github", {}).get("found", False) else "❌",
                        "LinkedIn": "✅" if performer.get("profiles", {}).get("linkedin", {}).get("found", False) else "❌"
                    })
                
                performers_df = pd.DataFrame(performers_data)
                st.dataframe(performers_df, use_container_width=True)
                
                # AI-Powered Insights
                st.markdown("---")
                st.markdown("### 🤖 AI-Powered Strategic Insights")
                
                if st.button("🧠 Generate Strategic Analysis", type="primary", key="generate_strategic_analysis"):
                    with st.spinner("🤖 Analyzing talent pool with AI..."):
                        insights_prompt = f"""
                        As a senior talent acquisition strategist, provide a comprehensive analysis of this talent pool:
                        
                        TALENT POOL OVERVIEW:
                        - Total candidates: {total_candidates}
                        - Average match score: {avg_score:.1f}%
                        - Excellent matches (90%+): {excellent_matches} ({(excellent_matches/total_candidates*100):.1f}%)
                        - Verified emails: {verified_emails} ({(verified_emails/total_candidates*100):.1f}%)
                        - GitHub profiles: {github_profiles} ({(github_profiles/total_candidates*100):.1f}%)
                        - LinkedIn profiles: {linkedin_profiles} ({(linkedin_profiles/total_candidates*100):.1f}%)
                        - Average technical score: {avg_tech_score:.1f}%
                        - Average experience score: {avg_exp:.1f}%
                        - Average education score: {avg_edu:.1f}%
                        
                        TOP SKILLS IN POOL:
                        {', '.join([skill for skill, _ in top_skills[:10]])}
                        
                        Provide insights on:
                        1. 🎯 TALENT QUALITY ASSESSMENT
                        2. 📋 HIRING STRATEGY RECOMMENDATIONS  
                        3. 🔍 MARKET POSITIONING INSIGHTS
                        4. ⚠️ POTENTIAL RISKS & MITIGATION
                        5. 🚀 COMPETITIVE ADVANTAGES
                        6. 📊 DIVERSITY & INCLUSION OBSERVATIONS
                        7. 🎯 ACTION ITEMS FOR RECRUITERS
                        
                        Format as a professional strategic report with clear sections and actionable recommendations.
                        """
                        
                        try:
                            strategic_insights = call_groq_api(insights_prompt, model="llama3-70b-8192")
                            
                            if isinstance(strategic_insights, dict) and "error" in strategic_insights:
                                st.error(f"Error generating insights: {strategic_insights['error']}")
                            else:
                                st.markdown(strategic_insights)
                                
                                # Download strategic report
                                st.markdown("---")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Create comprehensive report
                                    full_report = {
                                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "talent_pool_summary": {
                                            "total_candidates": total_candidates,
                                            "average_scores": {
                                                "overall": avg_score,
                                                "technical": avg_tech_score,
                                                "experience": avg_exp,
                                                "education": avg_edu
                                            },
                                            "verification_stats": {
                                                "verified_emails": verified_emails,
                                                "github_profiles": github_profiles,
                                                "linkedin_profiles": linkedin_profiles
                                            },
                                            "top_skills": dict(top_skills[:10])
                                        },
                                        "strategic_insights": strategic_insights,
                                        "top_performers": [
                                            {
                                                "name": p["resume_info"].get("name", "Unknown"),
                                                "scores": p["match_result"]["scores"],
                                                "email": p["resume_info"].get("email", "")
                                            } for p in top_performers[:5]
                                        ]
                                    }
                                    
                                    report_json = json.dumps(full_report, indent=2)
                                    b64_report = base64.b64encode(report_json.encode()).decode()
                                    download_link = f'<a href="data:application/json;base64,{b64_report}" download="strategic_talent_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json" class="btn" style="background-color:#4CAF50;color:white;padding:8px 12px;text-decoration:none;border-radius:4px;">📥 Download Strategic Report</a>'
                                    st.markdown(download_link, unsafe_allow_html=True)
                                
                                with col2:
                                    if st.button("📤 Share with Leadership", key="share_leadership"):
                                        # Create leadership summary email
                                        leadership_email = f"""Strategic Talent Pool Analysis
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
• Pool Size: {total_candidates} candidates
• Quality Score: {avg_score:.1f}% average match
• High-Quality Candidates: {excellent_matches} excellent matches ({(excellent_matches/total_candidates*100):.1f}%)
• Verification Rate: {verified_emails} verified profiles ({(verified_emails/total_candidates*100):.1f}%)

KEY FINDINGS:
{strategic_insights[:300]}...

Detailed strategic analysis and recommendations available in full report.

Best regards,
Talent Acquisition Team"""
                                        
                                        mailto_link = f"mailto:?subject=Strategic Talent Pool Analysis - {datetime.now().strftime('%Y-%m-%d')}&body={leadership_email.replace(chr(10), '%0D%0A')}"
                                        st.markdown(f"[📧 Email Leadership Team]({mailto_link})")
                        
                        except Exception as e:
                            st.error(f"Error generating strategic insights: {str(e)}")
                
            else:
                st.info("No candidates available for analysis")
        else:
            st.info("📋 Please complete the resume screening process first to view analytics")
            
            # Sample dashboard preview
            st.markdown("---")
            st.markdown("### 👁️ Dashboard Preview")
            st.info("This is what your analytics dashboard will look like once you've processed candidates:")
            
            # Mock metrics for preview
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Candidates", "---")
            with col2:
                st.metric("Average Match", "---%")
            with col3:
                st.metric("Excellent Matches", "---")
            with col4:
                st.metric("Verified Emails", "---%")
            with col5:
                st.metric("Avg Tech Score", "---%")

if __name__ == "__main__":
    main()