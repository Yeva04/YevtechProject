import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, ScrollView, Picker } from 'react-native';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [role, setRole] = useState('student');
  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [user, setUser] = useState(null);
  const [gender, setGender] = useState('Male');
  const [age, setAge] = useState('');
  const [attendanceRate, setAttendanceRate] = useState('');
  const [assignmentScore, setAssignmentScore] = useState('');
  const [quizScore, setQuizScore] = useState('');
  const [examScore, setExamScore] = useState('');
  const [studyHours, setStudyHours] = useState('');
  const [feedback, setFeedback] = useState('');
  const [accessToInternet, setAccessToInternet] = useState('YES');
  const [studySupportAtHome, setStudySupportAtHome] = useState('YES');
  const [studentId, setStudentId] = useState('');
  const [complaintFullName, setComplaintFullName] = useState('');
  const [complaintEmail, setComplaintEmail] = useState('');
  const [complaintPhone, setComplaintPhone] = useState('');
  const [complaintText, setComplaintText] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const API_URL = 'http://192.168.0.182:5000'; // Update to your local IP

  useEffect(() => {
    const checkLogin = async () => {
      const userData = await AsyncStorage.getItem('user');
      if (userData) {
        setUser(JSON.parse(userData));
        setIsLoggedIn(true);
      }
    };
    checkLogin();
  }, []);

  const handleLogin = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/login`, { identifier, password, role });
      if (response.data.success) {
        setUser(response.data);
        setIsLoggedIn(true);
        await AsyncStorage.setItem('user', JSON.stringify(response.data));
        setError('');
      } else {
        setError(response.data.error);
      }
    } catch (err) {
      setError('Login failed. Please try again.');
    }
  };

  const handlePredict = async () => {
    try {
      const data = {
        gender,
        age,
        attendance_rate: attendanceRate,
        assignment_score: assignmentScore,
        quiz_score: quizScore,
        exam_score: examScore,
        study_hours_per_week: studyHours,
        feedback,
        access_to_internet: accessToInternet,
        study_support_at_home: studySupportAtHome,
        student_id: role === 'admin' ? studentId : user.identifier
      };
      const response = await axios.post(`${API_URL}/api/predict`, data, {
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.data.success) {
        setPrediction(response.data.prediction);
        setMessage(response.data.message);
        setError('');
      } else {
        setError(response.data.error);
      }
    } catch (err) {
      setError('Prediction failed. Please check inputs.');
    }
  };

  const handleComplaint = async () => {
    try {
      const response = await axios.post(`${API_URL}/api/submit_complaint`, {
        full_name: complaintFullName,
        email: complaintEmail,
        phone_number: complaintPhone,
        complaint_text: complaintText
      }, {
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.data.success) {
        setError('');
        setMessage(response.data.message);
        setComplaintFullName('');
        setComplaintEmail('');
        setComplaintPhone('');
        setComplaintText('');
      } else {
        setError(response.data.error);
      }
    } catch (err) {
      setError('Complaint submission failed.');
    }
  };

  const handleLogout = async () => {
    await AsyncStorage.removeItem('user');
    setIsLoggedIn(false);
    setUser(null);
    setPrediction(null);
    setMessage('');
    setError('');
  };

  if (!isLoggedIn) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>Student Academic Predictor</Text>
        <Text style={styles.label}>Role:</Text>
        <Picker
          selectedValue={role}
          style={styles.picker}
          onValueChange={(value) => setRole(value)}
        >
          <Picker.Item label="Student" value="student" />
          <Picker.Item label="Admin" value="admin" />
        </Picker>
        <Text style={styles.label}>Identifier/Username:</Text>
        <TextInput style={styles.input} value={identifier} onChangeText={setIdentifier} />
        <Text style={styles.label}>Password:</Text>
        <TextInput style={styles.input} value={password} onChangeText={setPassword} secureTextEntry />
        <Button title="Login" onPress={handleLogin} />
        {error && <Text style={styles.error}>{error}</Text>}
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>{user.role === 'student' ? 'Student Dashboard' : 'Admin Dashboard'}</Text>
      <Button title="Logout" onPress={handleLogout} />
      <Text style={styles.label}>Gender:</Text>
      <Picker
        selectedValue={gender}
        style={styles.picker}
        onValueChange={(value) => setGender(value)}
      >
        <Picker.Item label="Male" value="Male" />
        <Picker.Item label="Female" value="Female" />
      </Picker>
      <Text style={styles.label}>Age (10-30):</Text>
      <TextInput style={styles.input} value={age} onChangeText={setAge} keyboardType="numeric" />
      <Text style={styles.label}>Attendance Rate (0-100):</Text>
      <TextInput style={styles.input} value={attendanceRate} onChangeText={setAttendanceRate} keyboardType="numeric" />
      <Text style={styles.label}>Assignment Score (0-100):</Text>
      <TextInput style={styles.input} value={assignmentScore} onChangeText={setAssignmentScore} keyboardType="numeric" />
      <Text style={styles.label}>Quiz Score (0-100):</Text>
      <TextInput style={styles.input} value={quizScore} onChangeText={setQuizScore} keyboardType="numeric" />
      <Text style={styles.label}>Exam Score (0-100):</Text>
      <TextInput style={styles.input} value={examScore} onChangeText={setExamScore} keyboardType="numeric" />
      <Text style={styles.label}>Study Hours per Week (0-168):</Text>
      <TextInput style={styles.input} value={studyHours} onChangeText={setStudyHours} keyboardType="numeric" />
      <Text style={styles.label}>Feedback:</Text>
      <TextInput style={styles.input} value={feedback} onChangeText={setFeedback} multiline />
      <Text style={styles.label}>Access to Internet:</Text>
      <Picker
        selectedValue={accessToInternet}
        style={styles.picker}
        onValueChange={(value) => setAccessToInternet(value)}
      >
        <Picker.Item label="YES" value="YES" />
        <Picker.Item label="NO" value="NO" />
      </Picker>
      <Text style={styles.label}>Study Support at Home:</Text>
      <Picker
        selectedValue={studySupportAtHome}
        style={styles.picker}
        onValueChange={(value) => setStudySupportAtHome(value)}
      >
        <Picker.Item label="YES" value="YES" />
        <Picker.Item label="NO" value="NO" />
      </Picker>
      {user.role === 'admin' && (
        <>
          <Text style={styles.label}>Student ID:</Text>
          <TextInput style={styles.input} value={studentId} onChangeText={setStudentId} />
        </>
      )}
      <Button title="Predict" onPress={handlePredict} />
      {prediction && (
        <>
          <Text style={styles.result}>Prediction: {prediction}</Text>
          <Text style={styles.result}>{message}</Text>
        </>
      )}
      {user.role === 'student' && (
        <>
          <Text style={styles.title}>Submit Complaint</Text>
          <Text style={styles.label}>Full Name:</Text>
          <TextInput style={styles.input} value={complaintFullName} onChangeText={setComplaintFullName} />
          <Text style={styles.label}>Email:</Text>
          <TextInput style={styles.input} value={complaintEmail} onChangeText={setComplaintEmail} />
          <Text style={styles.label}>Phone Number:</Text>
          <TextInput style={styles.input} value={complaintPhone} onChangeText={setComplaintPhone} />
          <Text style={styles.label}>Complaint:</Text>
          <TextInput style={styles.input} value={complaintText} onChangeText={setComplaintText} multiline />
          <Button title="Submit Complaint" onPress={handleComplaint} />
        </>
      )}
      {error && <Text style={styles.error}>{error}</Text>}
      {message && !prediction && <Text style={styles.result}>{message}</Text>}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: '#f0f8ff' },
  title: { fontSize: 24, fontWeight: 'bold', marginVertical: 10, color: '#4682b4' },
  label: { fontSize: 16, marginVertical: 5, color: '#5f9ea0' },
  input: { borderWidth: 1, padding: 10, marginBottom: 10, borderColor: '#4682b4', borderRadius: 5 },
  picker: { height: 50, marginBottom: 10 },
  error: { color: 'red', marginVertical: 10 },
  result: { fontSize: 18, marginVertical: 10, color: '#4682b4' }
});

export default App;