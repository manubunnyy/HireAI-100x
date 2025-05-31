"""
Email Handler Module
Handles automated email communications including offer letter generation and sending
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
import requests

class EmailHandler:
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587, sender_email: str = None, sender_password: str = None):
        """
        Initialize email handler
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender's email address
            sender_password: Sender's email password or app password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
    
    def generate_offer_letter_content(self, candidate_info: Dict[str, Any], job_details: Dict[str, Any], company_info: Dict[str, Any], ai_api_key: str = None) -> str:
        """
        Generate personalized offer letter content using AI
        """
        prompt = f"""
        Generate a professional and warm offer letter for a candidate based on the following information:
        
        Candidate Information:
        Name: {candidate_info.get('name', 'Candidate')}
        Position Applied: {job_details.get('position', 'Position')}
        Skills: {', '.join(candidate_info.get('skills', [])[:5])}
        
        Job Details:
        Position: {job_details.get('position', 'Position')}
        Department: {job_details.get('department', 'Department')}
        Salary Range: {job_details.get('salary_range', 'Competitive')}
        Start Date: {job_details.get('start_date', 'To be discussed')}
        
        Company Information:
        Company Name: {company_info.get('name', 'Company')}
        HR Contact: {company_info.get('hr_contact', 'HR Department')}
        
        Create a professional offer letter that includes:
        1. Warm congratulations
        2. Position details
        3. Key responsibilities (based on job requirements)
        4. Compensation and benefits overview
        5. Next steps
        6. Expression of excitement about them joining
        
        Make it personalized, professional, and engaging. Format it as a proper letter.
        """
        
        if ai_api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {ai_api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "llama3-70b-8192",
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                st.error(f"Error generating offer letter with AI: {str(e)}")
        
        # Fallback template
        return f"""
Dear {candidate_info.get('name', 'Candidate')},

We are delighted to inform you that you have been selected for the position of {job_details.get('position', 'Position')} at {company_info.get('name', 'our company')}.

After careful consideration of your application and interview performance, we believe you would be a valuable addition to our team.

Position Details:
- Title: {job_details.get('position', 'Position')}
- Department: {job_details.get('department', 'Department')}
- Reporting to: {job_details.get('reporting_to', 'To be communicated')}
- Start Date: {job_details.get('start_date', 'To be discussed')}

We were particularly impressed with your skills in {', '.join(candidate_info.get('skills', ['your technical expertise'])[:3])}.

Next Steps:
1. Please review this offer letter carefully
2. Contact our HR department if you have any questions
3. Confirm your acceptance by replying to this email

We look forward to welcoming you to our team!

Best regards,
{company_info.get('hr_contact', 'HR Department')}
{company_info.get('name', 'Company')}
"""
    
    def send_email(self, recipient_email: str, subject: str, body: str, attachments: List[str] = None) -> Dict[str, Any]:
        """
        Send an email with optional attachments
        
        Args:
            recipient_email: Recipient's email address
            subject: Email subject
            body: Email body (can be HTML)
            attachments: List of file paths to attach
            
        Returns:
            Dictionary with status and message
        """
        if not self.sender_email or not self.sender_password:
            return {"status": "error", "message": "Email credentials not configured"}
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachments if any
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {os.path.basename(file_path)}'
                            )
                            msg.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                text = msg.as_string()
                server.sendmail(self.sender_email, recipient_email, text)
            
            return {"status": "success", "message": f"Email sent successfully to {recipient_email}"}
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to send email: {str(e)}"}
    
    def send_bulk_emails(self, recipients: List[Dict[str, Any]], template: str, subject_template: str) -> List[Dict[str, Any]]:
        """
        Send personalized emails to multiple recipients
        
        Args:
            recipients: List of dictionaries with recipient info
            template: Email body template with placeholders
            subject_template: Subject template with placeholders
            
        Returns:
            List of results for each email
        """
        results = []
        
        for recipient in recipients:
            try:
                # Personalize subject and body
                subject = subject_template.format(**recipient)
                body = template.format(**recipient)
                
                # Send email
                result = self.send_email(
                    recipient.get('email'),
                    subject,
                    body,
                    recipient.get('attachments', [])
                )
                
                results.append({
                    "recipient": recipient.get('name', recipient.get('email')),
                    "status": result["status"],
                    "message": result["message"]
                })
                
            except Exception as e:
                results.append({
                    "recipient": recipient.get('name', recipient.get('email')),
                    "status": "error",
                    "message": str(e)
                })
        
        return results
    
    def create_email_template(self, template_type: str = "offer_letter") -> str:
        """
        Create email templates for different purposes
        """
        templates = {
            "offer_letter": """
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                            Congratulations, {name}! 🎉
                        </h2>
                        
                        <p>Dear {name},</p>
                        
                        <p>We are thrilled to extend an offer for the position of <strong>{position}</strong> at {company_name}.</p>
                        
                        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                            <h3 style="color: #2c3e50; margin-top: 0;">Offer Details:</h3>
                            <ul style="list-style-type: none; padding-left: 0;">
                                <li>📍 <strong>Position:</strong> {position}</li>
                                <li>🏢 <strong>Department:</strong> {department}</li>
                                <li>📅 <strong>Start Date:</strong> {start_date}</li>
                                <li>💰 <strong>Compensation:</strong> {salary_range}</li>
                            </ul>
                        </div>
                        
                        <p>{offer_details}</p>
                        
                        <div style="background-color: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0;">
                            <strong>Next Steps:</strong><br>
                            Please review this offer and respond within 5 business days. If you have any questions, 
                            don't hesitate to reach out to our HR team.
                        </div>
                        
                        <p>We look forward to having you join our team!</p>
                        
                        <p>Best regards,<br>
                        <strong>{hr_name}</strong><br>
                        {hr_title}<br>
                        {company_name}<br>
                        📧 {hr_email} | 📞 {hr_phone}</p>
                    </div>
                </body>
                </html>
            """,
            
            "interview_invitation": """
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <h2 style="color: #2c3e50;">Interview Invitation - {position}</h2>
                        
                        <p>Dear {name},</p>
                        
                        <p>Thank you for your interest in the {position} role at {company_name}. 
                        We were impressed with your application and would like to invite you for an interview.</p>
                        
                        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                            <h3 style="color: #2c3e50; margin-top: 0;">Interview Details:</h3>
                            <ul>
                                <li><strong>Date:</strong> {interview_date}</li>
                                <li><strong>Time:</strong> {interview_time}</li>
                                <li><strong>Format:</strong> {interview_format}</li>
                                <li><strong>Duration:</strong> {interview_duration}</li>
                            </ul>
                        </div>
                        
                        <p>Please confirm your availability by replying to this email.</p>
                        
                        <p>Best regards,<br>
                        {hr_name}<br>
                        {company_name}</p>
                    </div>
                </body>
                </html>
            """,
            
            "rejection_letter": """
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <h2 style="color: #2c3e50;">Thank You for Your Application</h2>
                        
                        <p>Dear {name},</p>
                        
                        <p>Thank you for taking the time to apply for the {position} position at {company_name} 
                        and for your interest in joining our team.</p>
                        
                        <p>After careful consideration, we have decided to move forward with other candidates 
                        whose qualifications more closely match our current needs.</p>
                        
                        <p>We were impressed by your background and encourage you to apply for future positions 
                        that match your skills and experience.</p>
                        
                        <p>We wish you the best in your job search and future endeavors.</p>
                        
                        <p>Sincerely,<br>
                        {hr_name}<br>
                        {company_name}</p>
                    </div>
                </body>
                </html>
            """
        }
        
        return templates.get(template_type, templates["offer_letter"]) 