import requests

session = requests.Session()
data = {
    'email': 'testing222@gmail.com',
    'username': 'testing222',
    'password': 'password123',
    'confirm_password': 'password123',
    'security_question': "What is your mother's last name?",
    'security_answer': 'smith'
}
response = session.post('http://127.0.0.1:5000/signup', data=data)
print("Status:", response.status_code)

if "sweetalert" in response.text or "alert" in response.text:
    import re
    alerts = re.findall(r'<div class="alert[^>]*>(.*?)</div>', response.text)
    print("Flashed Alerts:", alerts)
    
    # Also check for success
    if "Signup successful" in response.text:
        print("Success message found!")
