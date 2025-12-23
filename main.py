import os
import re
from db_src.DB_MODEL import save_image, get_images, getpath, \
    get_usertype, get_users, delete_user, get_username, create_user
from ai_grader.perform_classification import predict
from ai_grader.generate_caption import caption_g
from ai_grader.detect_opacity import opacity
from ai_grader.detect_external_devices import detect_devices
from ai_grader.generate_segmentation import segmentation
from ai_grader.generate_heatmaps import heatmap
from config import app
from routes import authRoutes
from flask_login import current_user, login_required,logout_user
from flask import Flask, request, render_template, logging, session, \
                                        redirect, url_for, flash, jsonify
APP_ROOT= os.path.dirname(os.path.abspath(__file__))

#LOGIN PAGE
@app.route('/login')
def login():
    """Handle user login and redirect based on user role."""
    if request.method == 'POST':
        session['email'] = request.form['email']
        User_Type = get_usertype(request.form['email'])
        if str(User_Type) == 'administrator':
            return redirect(url_for('AdminHome'))
        else:
            return redirect(url_for('home'))
    return render_template('login.html')

# SIGN-UP PAGE
@app.route('/signup')
def signup():
    """Render the sign-up page."""
    return render_template('signup.html')

# WELCOME PAGE
@app.route('/welcome')
def welcome():
    """Render the welcome page."""
    return render_template('welcome.html')

#CONTACT PAGE
@app.route('/contact')
#@login_required
def contact():
    """Render the contact information page."""
    return render_template('contact.html')

#ABOUT PAGE
@app.route('/about')
#@login_required
def about():
    """Render the about page."""
    return render_template('about.html')

#HOME PAGE
@app.route('/')
@app.route('/home')
@login_required
def home():
    """Render the home page for authorized users, displaying their uploaded images."""
    print(current_user)
    user_name=get_username(int(current_user.get_id()))
    any_uploads = ''
    any_uploads = get_images(int(current_user.get_id()))
    return render_template('home.html',  user_name=user_name, any_uploads=any_uploads)

@app.route('/AdminHome')
@login_required
def AdminHome():
    """Render the administrator home page, displaying user management information."""
    user_name = get_username(int(current_user.get_id()))
    print(current_user)
    any_uploads = ''
    any_uploads = get_users(int(current_user.get_id()))
    return render_template('AdminHome.html', user_name=user_name, any_uploads=any_uploads,)

@app.route("/upload", methods=['POST'])
def upload():
    """Handle patient image uploads and save them to the user's directory."""
    user_id = int(current_user.get_id())
    if not os.path.exists('static/Patient_images/User'+str(user_id)):
        os.makedirs('static/Patient_images/User'+str(user_id))
    target = os.path.join(APP_ROOT, 'static/Patient_images/User'+str(user_id))
    patient_name = request.form.get("name")
    any_uploads = get_images(int(current_user.get_id()))
    if not os.path.isdir(target):
        os.mkdir(target)
    filedata = request.files["file"]
    destination = "/".join([target, patient_name + '.jpg'])
    filedata.save(destination)
    if any_uploads:
        if(patient_name not in any_uploads):
            save_image(int(current_user.get_id()), str(patient_name), str(destination))
            return "success"
        else:
            return "updated"
    else:
        save_image(int(current_user.get_id()), str(patient_name), str(destination))
        return "success"

@app.route("/reg_doctor", methods=['GET', 'POST'])
def reg_doctor():
    """Register a new doctor in the system."""
    fname = request.form.get("fname")
    lname = request.form.get("lname")
    email = request.form.get("email")
    password = request.form.get("password")
    create_user(fname, lname, email, password, "docter")

    return 'success'

@app.route('/patient_name', methods=['GET', 'POST'])
def get_id():
    """Retrieve patient ID or redirect to the home page if not found."""
    p_name = request.form.get("patient_name")
    print(p_name)
    if p_name != 'False':
        return redirect(url_for("n1"))
    return render_template('home.html')

@app.route('/getdat')
def getdata():
    """Retrieve analysis data, including captions, predictions, and heatmaps for a patient image."""
    user_id = str(int(current_user.get_id()))
    p_name= request.args.get("p_name")
    path = getpath(p_name)
    path = ('User'+user_id+'/'+path)
    caption = caption_g(path, p_name)
    finding, classification = predict(path, p_name,user_id)
    heatmap(path, p_name, user_id)
    print(path)
    print(caption)
    return jsonify({"image_name": p_name, "path" : path, "caption" : caption, "finding" : finding})
@app.route('/getsegment')
def get_segment():
    """Perform image segmentation for a specific patient image."""
    user_id = str(int(current_user.get_id()))
    p_name = request.args.get("p_name")
    path = getpath(p_name)
    path = ('User' + user_id + '/' + path)
    segmentation(path, p_name, user_id)
    return jsonify({"image_name": p_name, "path" : path})

@app.route('/opacity')
def get_opacity():
    """Analyze a patient image for opacities."""
    user_id = str(int(current_user.get_id()))
    p_name = request.args.get("p_name")
    path = getpath(p_name)
    path = ('User' + user_id + '/' + path)
    opacity(path, p_name, user_id)
    return jsonify({"image_name": p_name, "path" : path})

@app.route('/external_devices')
def external_devices():
    """Detect external devices in a patient image."""
    user_id = str(int(current_user.get_id()))
    p_name = request.args.get("p_name")
    path = getpath(p_name)
    path = ('User' + user_id + '/' + path)
    detect_devices(path, p_name, user_id)
    return jsonify({"image_name": p_name, "path" : path})

@app.route('/get_full_report')
def get_full_report():
    """Generate a complete diagnostic report including all analyses for a patient image."""
    user_id = str(int(current_user.get_id()))
    p_name = request.args.get("p_name")
    path = getpath(p_name)
    path = ('User' + user_id + '/' + path)
    
    # Run all analyses
    caption = caption_g(path, p_name)
    finding, classification = predict(path, p_name, user_id)
    heatmap(path, p_name, user_id)
    segmentation(path, p_name, user_id)
    opacity(path, p_name, user_id)
    detect_devices(path, p_name, user_id)
    
    return jsonify({
        "image_name": p_name, 
        "path": path, 
        "caption": caption, 
        "finding": finding
    })
    
@app.route('/get_email')
def get_email():
    """Delete a user account using the provided email address."""
    u_email = request.args.get("u_email")
    delete_user(u_email)
    return 'ok'

@app.route('/logout')
def logout():
    """Log out the current user and redirect to the login page."""
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)