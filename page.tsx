"use client";

import React, { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import FaceRecognition from '../FaceRecognition';

const LoginPage = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirect = searchParams?.get('redirect') || '/';

  const [isLogin, setIsLogin] = useState(true); // 控制是登录还是注册模式
  const [isFaceVerification, setIsFaceVerification] = useState(false); // 控制是否显示人脸识别
  const [faceDetected, setFaceDetected] = useState(false); // 人脸是否已检测到
  const [faceImage, setFaceImage] = useState<string | null>(null); // 存储人脸图像
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    name: '',
    email: '',
    phone: '',
    role: 'user',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [capturedImageData, setCapturedImageData] = useState<string | null>(null);
  const [detectionMessage, setDetectionMessage] = useState('请将脸部对准摄像头'); // 检测状态消息

  // 处理表单字段变化
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  // 开始人脸验证流程
  const startFaceVerification = () => {
    console.log("开始人脸验证流程");
    setError('');
    setFaceDetected(false);
    setFaceImage(null);
    setCapturedImageData(null);
    setIsFaceVerification(true);
    setDetectionMessage('请将脸部对准摄像头，系统正在自动检测...');
  };

  // 处理人脸检测成功
  const handleFaceDetected = (imageData: string, detectedFaceImage: string | null) => {
    console.log("人脸检测成功回调");
    console.log("图像数据长度:", imageData.length);
    
    // 更新状态
    setCapturedImageData(imageData);
    setFaceImage(detectedFaceImage);
    setFaceDetected(true);
    
    // 延迟一秒后自动进行下一步操作
    setTimeout(() => {
      if (isLogin) {
        handleFaceLogin(imageData);
      } else {
        handleFaceRegistration(imageData);
      }
    }, 1000);
  };

  // 取消人脸验证
  const cancelFaceVerification = () => {
    console.log("取消人脸验证");
    setIsFaceVerification(false);
    setLoading(false);
    setFaceDetected(false);
    setFaceImage(null);
    setCapturedImageData(null);
  };

  // 管理员直接注册（不进行人脸验证）
  const handleAdminRegister = async () => {
    console.log("开始管理员直接注册");
    setLoading(true);
    
    try {
      // 1. 先进行管理员注册
      console.log("发送管理员注册请求");
      const registerResponse = await fetch('/api/auth/register/admin', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      console.log("注册响应状态:", registerResponse.status);
      const registerResult = await registerResponse.json();
      
      if (!registerResponse.ok) {
        console.error("注册失败:", registerResult.message);
        setError(registerResult.message || '注册失败，请稍后再试');
        setLoading(false);
        return;
      }
      
      console.log("管理员注册成功，准备登录");
      
      // 2. 登录获取token
      const loginResponse = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: formData.username,
          password: formData.password,
          role: 'admin'
        }),
      });
      
      console.log("登录响应状态:", loginResponse.status);
      const loginResult = await loginResponse.json();
      
      if (!loginResponse.ok) {
        console.error("登录失败:", loginResult.message);
        setError(loginResult.message || '登录失败，请稍后再试');
        setLoading(false);
        return;
      }
      
      console.log("管理员登录成功");
      
      // 保存登录信息
      localStorage.setItem('currentRole', 'admin');
      localStorage.setItem('isLoggedIn', 'true');
      localStorage.setItem('userId', loginResult.user.id.toString());
      localStorage.setItem('token', loginResult.token);
      
      // 重定向到管理员仪表盘
      console.log("重定向到管理员仪表盘");
      router.push('/admin/dashboard');
      
    } catch (err) {
      console.error('管理员注册请求失败:', err);
      setError('注册请求失败，请稍后再试');
      setLoading(false);
    }
  };

  // 使用人脸图像进行登录
  const handleFaceLogin = async (imageData?: string) => {
    console.log("开始人脸登录");
    
    // 使用传入的参数或当前状态
    const dataToUse = imageData || capturedImageData;
    
    if (!dataToUse) {
      console.error("没有捕获的人脸图像数据");
      setError("人脸图像数据缺失，请重新检测");
      return;
    }
    
    console.log("使用的图像数据长度:", dataToUse.length);
    setLoading(true);
    setDetectionMessage('正在进行人脸登录验证...');
    setError('');
    
    try {
      console.log("发送人脸登录请求");
      const response = await fetch('/api/auth/face', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ face_data: dataToUse }),
      });
      
      console.log("人脸登录响应状态:", response.status);
      const result = await response.json();
      
      if (response.ok) {
        console.log("人脸登录成功");
        setDetectionMessage('登录成功，正在跳转...');
        
        // 登录成功
        localStorage.setItem('currentRole', result.role);
        localStorage.setItem('isLoggedIn', 'true');
        localStorage.setItem('userId', result.user.id.toString());
        localStorage.setItem('token', result.token);
        
        // 根据角色重定向
        let redirectPath = redirect;
        if (result.role === 'admin') {
          redirectPath = '/admin/dashboard';
        }
        
        console.log("重定向到:", redirectPath);
        router.push(redirectPath);
      } else {
        console.error("人脸登录失败:", result.message);
        setError(result.message || '人脸登录失败，请使用账号密码登录');
        setDetectionMessage('验证失败，请重试...');
      }
    } catch (err) {
      console.error('人脸登录请求失败:', err);
      setError('人脸登录请求失败，请稍后再试');
      setDetectionMessage('网络错误，请重试...');
    } finally {
      setLoading(false);
    }
  };

  // 注册并绑定人脸 
  const handleFaceRegistration = async (imageData?: string) => {
    console.log("开始用户注册并绑定人脸");
    
    // 使用传入的参数或当前状态
    const dataToUse = imageData || capturedImageData;
    
    if (!dataToUse) {
      console.error("没有捕获的人脸图像数据");
      setError("人脸图像数据缺失，请重新检测");
      return;
    }
    
    console.log("使用的图像数据长度:", dataToUse.length);
    setLoading(true);
    setDetectionMessage('正在注册并绑定人脸...');
    setError('');
    
    try {
      // 1. 先进行用户注册
      console.log("发送用户注册请求");
      const registerResponse = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      console.log("注册响应状态:", registerResponse.status);
      const registerResult = await registerResponse.json();
      
      if (!registerResponse.ok) {
        console.error("注册失败:", registerResult.message);
        setError(registerResult.message || '注册失败，请稍后再试');
        setDetectionMessage('注册失败，请检查信息后重试...');
        return;
      }
      
      console.log("用户注册成功，准备登录获取token");
      setDetectionMessage('注册成功，正在登录...');
      
      // 2. 登录获取token
      const loginResponse = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: formData.username,
          password: formData.password,
          role: formData.role
        }),
      });
      
      console.log("登录响应状态:", loginResponse.status);
      const loginResult = await loginResponse.json();
      
      if (!loginResponse.ok) {
        console.error("登录失败:", loginResult.message);
        setError(loginResult.message || '登录失败，请稍后再试');
        setDetectionMessage('登录失败，请稍后重试...');
        return;
      }
      
      console.log("登录成功，准备绑定人脸");
      setDetectionMessage('登录成功，正在绑定人脸...');
      
      // 3. 使用token进行人脸绑定
      const faceResponse = await fetch('/api/users/face-verification', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${loginResult.token}`
        },
        body: JSON.stringify({ face_data: dataToUse }),
      });
      
      console.log("人脸绑定响应状态:", faceResponse.status);
      const faceResult = await faceResponse.json();
      
      if (faceResponse.ok) {
        console.log("人脸绑定成功");
        setDetectionMessage('人脸绑定成功，正在跳转...');
        
        // 人脸绑定成功，保存登录信息
        localStorage.setItem('currentRole', loginResult.role);
        localStorage.setItem('isLoggedIn', 'true');
        localStorage.setItem('userId', loginResult.user.id.toString());
        localStorage.setItem('token', loginResult.token);
        
        // 根据角色重定向
        let redirectPath = redirect;
        if (loginResult.role === 'admin') {
          redirectPath = '/admin/dashboard';
        }
        
        console.log("重定向到:", redirectPath);
        router.push(redirectPath);
      } else {
        console.error("人脸绑定失败:", faceResult.message);
        setError(faceResult.message || '人脸绑定失败，请稍后再试');
        setDetectionMessage('人脸绑定失败，请重试...');
      }
    } catch (err) {
      console.error('注册或人脸绑定请求失败:', err);
      setError('注册或人脸绑定请求失败，请稍后再试');
      setDetectionMessage('网络错误，请重试...');
    } finally {
      setLoading(false);
    }
  };

  // 常规登录处理
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log("开始常规登录");
    setError('');
    setLoading(true);

    try {
      console.log("发送登录请求");
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: formData.username,
          password: formData.password,
          role: formData.role
        }),
      });
      
      console.log("登录响应状态:", response.status);
      const result = await response.json();
      
      if (response.ok) {
        // 如果是管理员角色，直接登录成功
        if (formData.role === 'admin') {
          console.log("管理员登录成功，直接跳转");
          localStorage.setItem('currentRole', result.role);
          localStorage.setItem('isLoggedIn', 'true');
          localStorage.setItem('userId', result.user.id.toString());
          localStorage.setItem('token', result.token);
          router.push('/admin/dashboard');
        } else {
          // 如果是普通用户，需要进行人脸验证
          console.log("普通用户登录成功，准备开始人脸验证");
          localStorage.setItem('tempToken', result.token);
          startFaceVerification();
        }
      } else {
        console.error("登录失败:", result.message);
        setError(result.message || '登录失败，请检查用户名和密码');
        setLoading(false);
      }
    } catch (err) {
      console.error('登录请求失败:', err);
      setError('登录请求失败，请稍后再试');
      setLoading(false);
    }
  };

  // 注册处理 - 修改后判断角色
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log("开始注册流程");
    setError('');
    setLoading(true);

    // 验证表单
    if (!formData.username || !formData.password || !formData.name) {
      console.error("表单验证失败: 缺少必填字段");
      setError('用户名、密码和姓名为必填项');
      setLoading(false);
      return;
    }

    // 根据角色选择注册流程
    if (formData.role === 'admin') {
      // 管理员直接注册，无需人脸验证
      console.log("管理员注册，跳过人脸验证");
      handleAdminRegister();
    } else {
      // 普通用户需要进行人脸验证
      console.log("普通用户注册，开始人脸验证");
      startFaceVerification();
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-blue-50">
      <div className="w-full max-w-md rounded-lg bg-white p-8 shadow-md">
        <div className="mb-6 text-center">
          <h2 className="text-2xl font-bold text-blue-700">景区票务系统</h2>
          <p className="text-gray-600">便捷购票，轻松游玩</p>
        </div>
        
        {!isFaceVerification ? (
          <>
            <div className="mb-6 flex">
              <button
                onClick={() => setIsLogin(true)}
                className={`flex-1 py-2 font-medium ${isLogin ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-500'}`}
              >
                登录
              </button>
              <button
                onClick={() => setIsLogin(false)}
                className={`flex-1 py-2 font-medium ${!isLogin ? 'border-b-2 border-blue-500 text-blue-500' : 'text-gray-500'}`}
              >
                注册
              </button>
            </div>
            
            {error && (
              <div className="mb-4 rounded bg-red-100 p-3 text-red-700">
                {error}
              </div>
            )}
            
            <form onSubmit={isLogin ? handleLogin : handleRegister}>
              <div className="mb-4">
                <label className="mb-2 block text-sm font-bold text-gray-700" htmlFor="role">
                  选择角色
                </label>
                <select
                  id="role"
                  name="role"
                  value={formData.role}
                  onChange={handleChange}
                  className="w-full appearance-none rounded border px-3 py-2 leading-tight text-gray-700 shadow focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="user">游客</option>
                  <option value="admin">管理员</option>
                </select>
              </div>
              
              <div className="mb-4">
                <label className="mb-2 block text-sm font-bold text-gray-700" htmlFor="username">
                  用户名
                </label>
                <input
                  type="text"
                  id="username"
                  name="username"
                  value={formData.username}
                  onChange={handleChange}
                  className="w-full appearance-none rounded border px-3 py-2 leading-tight text-gray-700 shadow focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="输入用户名"
                  required
                />
              </div>
              
              <div className="mb-4">
                <label className="mb-2 block text-sm font-bold text-gray-700" htmlFor="password">
                  密码
                </label>
                <input
                  type="password"
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className="w-full appearance-none rounded border px-3 py-2 leading-tight text-gray-700 shadow focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="输入密码"
                  required
                />
              </div>
              
              {!isLogin && (
                <>
                  <div className="mb-4">
                    <label className="mb-2 block text-sm font-bold text-gray-700" htmlFor="name">
                      姓名
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleChange}
                      className="w-full appearance-none rounded border px-3 py-2 leading-tight text-gray-700 shadow focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="输入姓名"
                      required
                    />
                  </div>
                  
                  <div className="mb-4">
                    <label className="mb-2 block text-sm font-bold text-gray-700" htmlFor="email">
                      电子邮箱 (选填)
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      className="w-full appearance-none rounded border px-3 py-2 leading-tight text-gray-700 shadow focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="输入电子邮箱"
                    />
                  </div>
                  
                  <div className="mb-4">
                    <label className="mb-2 block text-sm font-bold text-gray-700" htmlFor="phone">
                      手机号码 (选填)
                    </label>
                    <input
                      type="tel"
                      id="phone"
                      name="phone"
                      value={formData.phone}
                      onChange={handleChange}
                      className="w-full appearance-none rounded border px-3 py-2 leading-tight text-gray-700 shadow focus:outline-none focus:ring-2 focus:ring-blue-500"
                      placeholder="输入手机号码"
                    />
                  </div>
                </>
              )}
              
              <div className="flex items-center justify-between">
                <button
                  type="submit"
                  disabled={loading}
                  className="rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-300"
                >
                  {loading ? '处理中...' : isLogin ? '登录' : '注册'}
                </button>
                
                <div>
                  <p className="text-xs text-gray-500">
                    {isLogin ? '没有账号？' : '已有账号？'}
                    <button 
                      type="button"
                      className="text-blue-500 hover:underline"
                      onClick={() => setIsLogin(!isLogin)}
                    >
                      {isLogin ? '注册新账号' : '立即登录'}
                    </button>
                  </p>
                </div>
              </div>
            </form>
          </>
        ) : (
          <FaceRecognition
            onFaceDetected={handleFaceDetected}
            onCancel={cancelFaceVerification}
            loading={loading}
            autoDetectionEnabled={true}
            detectionMessage={detectionMessage}
            error={error}
            faceDetected={faceDetected}
            faceImage={faceImage}
          />
        )}
      </div>
    </div>
  );
};

export default LoginPage;