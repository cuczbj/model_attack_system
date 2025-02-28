"use client";

import React, { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Paper from "@mui/material/Paper";
import Chip from "@mui/material/Chip";
import Button from "@mui/material/Button";
import IconButton from "@mui/material/IconButton";
import LinearProgress from "@mui/material/LinearProgress";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Grid from "@mui/material/Grid";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Divider from "@mui/material/Divider";
import CircularProgress from "@mui/material/CircularProgress";
import Alert from "@mui/material/Alert";
// 模拟数据 - 实际应用中应从API获取
const mockTasks = [
  {
    id: "task-001",
    name: "MNIST模型反演攻击",
    model: "MNIST分类器",
    attack_type: "模型反演",
    status: "running",
    progress: 65,
    create_time: "2025-02-26 08:30:45",
    start_time: "2025-02-26 08:31:12",
    estimated_time_left: "14分钟",
    description: "针对MNIST手写数字识别模型的反演攻击实验"
  },
  {
    id: "task-002",
    name: "人脸识别模型GAN攻击",
    model: "FaceNet变体",
    attack_type: "GAN辅助重建",
    status: "completed",
    progress: 100,
    create_time: "2025-02-25 14:20:33",
    start_time: "2025-02-25 14:22:01",
    end_time: "2025-02-25 16:45:22",
    description: "使用GAN辅助技术重建人脸识别模型的训练数据"
  },
  {
    id: "task-003",
    name: "医疗数据成员推理",
    model: "疾病诊断模型",
    attack_type: "成员推理",
    status: "failed",
    progress: 27,
    create_time: "2025-02-24 09:15:07",
    start_time: "2025-02-24 09:16:32",
    end_time: "2025-02-24 09:34:18",
    error_message: "GPU内存不足",
    description: "评估医疗诊断模型的成员推理攻击抵抗力"
  },
  {
    id: "task-004",
    name: "ResNet50梯度泄露测试",
    model: "ResNet50-ImageNet",
    attack_type: "梯度泄露",
    status: "queued",
    progress: 0,
    create_time: "2025-02-26 10:05:39",
    description: "通过梯度信息重建ImageNet训练样本"
  },
  {
    id: "task-005",
    name: "语言模型知识提取",
    model: "BERT变体",
    attack_type: "模型反演",
    status: "paused",
    progress: 43,
    create_time: "2025-02-25 22:10:27",
    start_time: "2025-02-25 22:12:05",
    description: "从语言模型中提取潜在的训练数据知识"
  }
];

// 状态映射
const statusMap = {
  queued: { label: "排队中", color: "default" },
  running: { label: "运行中", color: "primary" },
  paused: { label: "已暂停", color: "warning" },
  completed: { label: "已完成", color: "success" },
  failed: { label: "失败", color: "error" }
};

// 资源使用模拟数据
const resourceUsage = {
  cpu: 75,
  memory: 60,
  gpu: 90,
  disk: 45
};

// 任务详情对话框内容
const TaskDetailDialog = ({ open, task, onClose }) => {
  const [activeTab, setActiveTab] = useState(0);

  if (!task) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        任务详情: {task.name}
        <Typography variant="subtitle2" color="text.secondary">
          ID: {task.id}
        </Typography>
      </DialogTitle>
      <DialogContent dividers>
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
          aria-label="task details tabs"
        >
          <Tab label="基本信息" />
          <Tab label="参数配置" />
          <Tab label="执行日志" />
          <Tab label="性能监控" />
        </Tabs>
        <Box sx={{ p: 2 }}>
          {activeTab === 0 && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="subtitle1">任务描述</Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {task.description}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">目标模型</Typography>
                <Typography variant="body2">{task.model}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">攻击类型</Typography>
                <Typography variant="body2">{task.attack_type}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">创建时间</Typography>
                <Typography variant="body2">{task.create_time}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">开始时间</Typography>
                <Typography variant="body2">{task.start_time || "尚未开始"}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">完成时间</Typography>
                <Typography variant="body2">{task.end_time || "尚未完成"}</Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="subtitle2">状态</Typography>
                <Chip 
                  label={statusMap[task.status].label} 
                  color={statusMap[task.status].color} 
                  size="small" 
                />
              </Grid>
              {task.status === "running" && (
                <Grid item xs={12}>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      进度: {task.progress}% (预计剩余时间: {task.estimated_time_left})
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={task.progress} 
                      sx={{ mt: 1, mb: 1 }} 
                    />
                  </Box>
                </Grid>
              )}
              {task.status === "failed" && (
                <Grid item xs={12}>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" color="error">
                      错误信息
                    </Typography>
                    <Paper sx={{ p: 1, bgcolor: "#FFF4F4", mt: 1 }}>
                      <Typography variant="body2" color="error">
                        {task.error_message}
                      </Typography>
                    </Paper>
                  </Box>
                </Grid>
              )}
            </Grid>
          )}

          {activeTab === 1 && (
            <Box>
              <Typography variant="h6" sx={{ mb: 2 }}>
                攻击参数配置
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">分类数</Typography>
                    <Typography variant="body1">10</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">迭代次数</Typography>
                    <Typography variant="body1">1000</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">学习率</Typography>
                    <Typography variant="body1">0.01</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">批次大小</Typography>
                    <Typography variant="body1">64</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">正则化参数</Typography>
                    <Typography variant="body1">0.0001</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">随机种子</Typography>
                    <Typography variant="body1">42</Typography>
                  </Grid>
                </Grid>
              </Paper>
              
              <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
                模型配置
              </Typography>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">模型类型</Typography>
                    <Typography variant="body1">卷积神经网络</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">模型版本</Typography>
                    <Typography variant="body1">v2.3.1</Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2">模型访问方式</Typography>
                    <Typography variant="body1">本地文件</Typography>
                  </Grid>
                </Grid>
              </Paper>
            </Box>
          )}

          {activeTab === 2 && (
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                执行日志
              </Typography>
              <Paper
                sx={{
                  p: 1.5,
                  bgcolor: "#f5f5f5",
                  fontFamily: "monospace",
                  fontSize: "0.875rem",
                  height: "300px",
                  overflow: "auto"
                }}
              >
                {task.status === "running" && (
                  <>
                    <div>[2025-02-26 08:31:12] 任务启动</div>
                    <div>[2025-02-26 08:31:14] 加载目标模型: {task.model}</div>
                    <div>[2025-02-26 08:31:20] 初始化攻击环境</div>
                    <div>[2025-02-26 08:31:25] 开始 {task.attack_type} 攻击</div>
                    <div>[2025-02-26 08:35:40] 迭代进度: 10%</div>
                    <div>[2025-02-26 08:40:15] 迭代进度: 20%</div>
                    <div>[2025-02-26 08:45:02] 迭代进度: 30%</div>
                    <div>[2025-02-26 08:50:18] 迭代进度: 40%</div>
                    <div>[2025-02-26 08:55:33] 迭代进度: 50%</div>
                    <div>[2025-02-26 09:00:47] 迭代进度: 60%</div>
                    <div>[2025-02-26 09:05:02] 当前进度: 65%</div>
                    <div>[2025-02-26 09:05:03] 中间结果保存成功</div>
                  </>
                )}
                {task.status === "completed" && (
                  <>
                    <div>[2025-02-25 14:22:01] 任务启动</div>
                    <div>[2025-02-25 14:22:05] 加载目标模型: {task.model}</div>
                    <div>[2025-02-25 14:22:15] 初始化攻击环境</div>
                    <div>[2025-02-25 14:22:30] 开始 {task.attack_type} 攻击</div>
                    <div>[2025-02-25 14:30:40] 迭代进度: 10%</div>
                    <div>[2025-02-25 14:45:15] 迭代进度: 25%</div>
                    <div>[2025-02-25 15:15:02] 迭代进度: 50%</div>
                    <div>[2025-02-25 15:45:18] 迭代进度: 75%</div>
                    <div>[2025-02-25 16:15:33] 迭代进度: 90%</div>
                    <div>[2025-02-25 16:40:47] 迭代进度: 100%</div>
                    <div>[2025-02-25 16:44:02] 生成最终结果</div>
                    <div>[2025-02-25 16:45:03] 结果保存成功</div>
                    <div>[2025-02-25 16:45:22] 任务完成</div>
                  </>
                )}
                {task.status === "failed" && (
                  <>
                    <div>[2025-02-24 09:16:32] 任务启动</div>
                    <div>[2025-02-24 09:16:35] 加载目标模型: {task.model}</div>
                    <div>[2025-02-24 09:16:40] 初始化攻击环境</div>
                    <div>[2025-02-24 09:17:00] 开始 {task.attack_type} 攻击</div>
                    <div>[2025-02-24 09:20:40] 迭代进度: 10%</div>
                    <div>[2025-02-24 09:25:15] 迭代进度: 20%</div>
                    <div>[2025-02-24 09:30:02] 迭代进度: 27%</div>
                    <div style={{ color: "red" }}>
                      [2025-02-24 09:34:15] 错误: CUDA out of memory. Tried to allocate 2.20 GiB
                    </div>
                    <div style={{ color: "red" }}>
                      [2025-02-24 09:34:18] 任务失败: GPU内存不足
                    </div>
                  </>
                )}
              </Paper>
            </Box>
          )}

          {activeTab === 3 && (
            <Box>
              <Typography variant="h6" sx={{ mb: 2 }}>
                资源使用情况
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2">CPU使用率</Typography>
                      <LinearProgress
                        variant="determinate"
                        value={resourceUsage.cpu}
                        sx={{ mt: 2, mb: 1 }}
                      />
                      <Typography variant="body2" sx={{ textAlign: "right" }}>
                        {resourceUsage.cpu}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2">内存使用率</Typography>
                      <LinearProgress
                        variant="determinate"
                        value={resourceUsage.memory}
                        sx={{ mt: 2, mb: 1 }}
                      />
                      <Typography variant="body2" sx={{ textAlign: "right" }}>
                        {resourceUsage.memory}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2">GPU使用率</Typography>
                      <LinearProgress
                        variant="determinate"
                        value={resourceUsage.gpu}
                        color="secondary"
                        sx={{ mt: 2, mb: 1 }}
                      />
                      <Typography variant="body2" sx={{ textAlign: "right" }}>
                        {resourceUsage.gpu}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle2">磁盘I/O</Typography>
                      <LinearProgress
                        variant="determinate"
                        value={resourceUsage.disk}
                        color="success"
                        sx={{ mt: 2, mb: 1 }}
                      />
                      <Typography variant="body2" sx={{ textAlign: "right" }}>
                        {resourceUsage.disk}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions>
        {task.status === "running" && (
          <>
            <Button color="warning">暂停</Button>
            <Button color="error">终止</Button>
          </>
        )}
        {task.status === "paused" && (
          <Button color="primary">恢复</Button>
        )}
        {task.status === "failed" && (
          <Button color="primary">重试</Button>
        )}
        {task.status === "completed" && (
          <Button color="primary">查看结果</Button>
        )}
        <Button onClick={onClose}>关闭</Button>
      </DialogActions>
    </Dialog>
  );
};
// 系统资源占用组件
const SystemResourcesMonitor = () => {
  const [resources, setResources] = useState({
    cpu: 0,
    memory: 0,
    gpu: 0,
    gpu_memory: 0,
    disk: 0,
    network: 0,
    timestamp: null
  });
  
  useEffect(() => {
    // Connect to the server-sent events endpoint
    const eventSource = new EventSource('http://127.0.0.1:5000/system-metrics');
    
    // Handle incoming events
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setResources(data);
    };
    
    // Handle connection error
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
    };
    
    // Clean up on component unmount
    return () => {
      eventSource.close();
    };
  }, []);
  
  return (
    <Paper 
      elevation={2} 
      sx={{ 
        p: 2, 
        mb: 3, 
        borderRadius: 2,
        background: 'linear-gradient(to right, #f5f7fa, #e4e7eb)'
      }}
    >
      <Typography variant="h6" gutterBottom sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
        系统资源监控
        <Chip 
          label="实时" 
          color="success" 
          size="small" 
          sx={{ ml: 2, height: 20 }} 
        />
      </Typography>
      
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6} md={4}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">GPU占用率</Typography>
              <Typography variant="body2" fontWeight="bold" color={resources.gpu > 90 ? "error.main" : "text.primary"}>
                {Math.round(resources.gpu)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={resources.gpu} 
              sx={{ 
                mt: 1, 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'rgba(144, 202, 249, 0.3)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: resources.gpu > 90 ? '#f44336' : (resources.gpu > 75 ? '#ff9800' : '#2196f3'),
                },
              }} 
            />
          </Box>
        </Grid>
        
        <Grid item xs={12} sm={6} md={4}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">GPU内存</Typography>
              <Typography variant="body2" fontWeight="bold" color={resources.gpu_memory > 90 ? "error.main" : "text.primary"}>
                {Math.round(resources.gpu_memory)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={resources.gpu_memory} 
              sx={{ 
                mt: 1, 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'rgba(156, 39, 176, 0.3)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: resources.gpu_memory > 90 ? '#f44336' : (resources.gpu_memory > 75 ? '#ff9800' : '#9c27b0'),
                },
              }} 
            />
          </Box>
        </Grid>
        
        <Grid item xs={12} sm={6} md={4}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">CPU占用率</Typography>
              <Typography variant="body2" fontWeight="bold" color={resources.cpu > 90 ? "error.main" : "text.primary"}>
                {Math.round(resources.cpu)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={resources.cpu} 
              sx={{ 
                mt: 1, 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'rgba(76, 175, 80, 0.3)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: resources.cpu > 90 ? '#f44336' : (resources.cpu > 75 ? '#ff9800' : '#4caf50'),
                },
              }} 
            />
          </Box>
        </Grid>
        
        <Grid item xs={12} sm={6} md={4}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">内存使用率</Typography>
              <Typography variant="body2" fontWeight="bold" color={resources.memory > 90 ? "error.main" : "text.primary"}>
                {Math.round(resources.memory)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={resources.memory} 
              sx={{ 
                mt: 1, 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'rgba(255, 152, 0, 0.3)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: resources.memory > 90 ? '#f44336' : (resources.memory > 75 ? '#ff9800' : '#ff9800'),
                },
              }} 
            />
          </Box>
        </Grid>
        
        <Grid item xs={12} sm={6} md={4}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">磁盘I/O</Typography>
              <Typography variant="body2" fontWeight="bold">
                {Math.round(resources.disk)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={resources.disk} 
              sx={{ 
                mt: 1, 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'rgba(33, 150, 243, 0.3)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: '#3f51b5',
                },
              }} 
            />
          </Box>
        </Grid>
        
        <Grid item xs={12} sm={6} md={4}>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">网络带宽</Typography>
              <Typography variant="body2" fontWeight="bold">
                {Math.round(resources.network)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={resources.network} 
              sx={{ 
                mt: 1, 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'rgba(233, 30, 99, 0.3)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: '#e91e63',
                },
              }} 
            />
          </Box>
        </Grid>
      </Grid>
      
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
        <Typography variant="caption" color="text.secondary">
          上次更新: {resources.timestamp ? new Date(resources.timestamp * 1000).toLocaleTimeString() : '等待数据...'}
        </Typography>
      </Box>
    </Paper>
  );
};

const TaskStatus = () => {
  const [tasks, setTasks] = useState([]);
  const [selectedTask, setSelectedTask] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchTasks = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://127.0.0.1:5000/api/tasks?status=${filter}`);
      if (!response.ok) {
        throw new Error("获取任务列表失败");
      }
      const data = await response.json();
      setTasks(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTasks();
    
    // 设置定时器每10秒刷新一次数据
    const interval = setInterval(() => {
      fetchTasks();
    }, 10000);
    
    return () => clearInterval(interval);
  }, [filter]);

  const handleViewDetails = async (task) => {
    try {
      const response = await fetch(`http://127.0.0.1:5000/api/tasks/${task.id}`);
      if (!response.ok) {
        throw new Error("获取任务详情失败");
      }
      const taskDetail = await response.json();
      setSelectedTask(taskDetail);
      setDialogOpen(true);
    } catch (err) {
      console.error("Error fetching task details:", err);
    }
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
  };

  const handleTaskAction = async (taskId, action) => {
    try {
      // 根据操作类型更新任务状态
      let newStatus;
      let errorMessage = '';
      
      switch (action) {
        case 'start':
          newStatus = 'running';
          break;
        case 'pause':
          newStatus = 'paused';
          break;
        case 'resume':
          newStatus = 'running';
          break;
        case 'cancel':
        case 'terminate':
          newStatus = 'failed';
          errorMessage = action === 'cancel' ? '任务已取消' : '任务已终止';
          break;
        case 'retry':
          newStatus = 'queued';
          break;
        case 'delete':
          // 处理删除任务的逻辑
          const deleteResponse = await fetch(`http://127.0.0.1:5000/api/tasks/${taskId}`, {
            method: 'DELETE'
          });
          if (!deleteResponse.ok) {
            throw new Error("删除任务失败");
          }
          fetchTasks();
          return;
        default:
          return;
      }

      const response = await fetch(`http://127.0.0.1:5000/api/tasks/${taskId}/status`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          status: newStatus,
          error_message: errorMessage
        }),
      });

      if (!response.ok) {
        throw new Error("更新任务状态失败");
      }

      // 更新当前任务列表中的状态
      setTasks(prevTasks => 
        prevTasks.map(task => 
          task.id === taskId 
            ? { ...task, status: newStatus, error_message: errorMessage || task.error_message } 
            : task
        )
      );

      // 如果当前正在查看该任务的详情，也更新详情
      if (selectedTask && selectedTask.id === taskId) {
        setSelectedTask(prev => ({
          ...prev,
          status: newStatus,
          error_message: errorMessage || prev.error_message
        }));
      }
    } catch (err) {
      console.error("Error updating task:", err);
    }
  };

  const handleCreateTask = () => {
    // 导航到结果页面
    window.location.href = '/results';
  };

  const handleRefresh = () => {
    fetchTasks();
  };

  const filteredTasks = filter === "all" 
    ? tasks 
    : tasks.filter(task => task.status === filter);

  const getStatusActions = (task) => {
    switch (task.status) {
      case "queued":
        return (
          <>
            <Button size="small" color="primary" onClick={() => handleTaskAction(task.id, 'start')}>
              启动
            </Button>
            <Button size="small" color="error" onClick={() => handleTaskAction(task.id, 'cancel')}>
              取消
            </Button>
          </>
        );
      case "running":
        return (
          <>
            <Button size="small" color="warning" onClick={() => handleTaskAction(task.id, 'pause')}>
              暂停
            </Button>
            <Button size="small" color="error" onClick={() => handleTaskAction(task.id, 'terminate')}>
              终止
            </Button>
          </>
        );
      case "paused":
        return (
          <>
            <Button size="small" color="primary" onClick={() => handleTaskAction(task.id, 'resume')}>
              恢复
            </Button>
            <Button size="small" color="error" onClick={() => handleTaskAction(task.id, 'terminate')}>
              终止
            </Button>
          </>
        );
      case "completed":
        return (
          <Button size="small" color="primary" onClick={() => window.location.href = `/results?task=${task.id}`}>
            查看结果
          </Button>
        );
      case "failed":
        return (
          <>
            <Button size="small" color="primary" onClick={() => handleTaskAction(task.id, 'retry')}>
              重试
            </Button>
            <Button size="small" color="error" onClick={() => handleTaskAction(task.id, 'delete')}>
              删除
            </Button>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        任务状态
      </Typography>
      <Typography variant="body1" paragraph>
        监控和管理攻击任务的执行状态、进度和结果。
      </Typography>

      <SystemResourcesMonitor />
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ mb: 3 }}>
        <Tabs
          value={filter}
          onChange={(e, newValue) => setFilter(newValue)}
          aria-label="任务状态筛选"
        >
          <Tab value="all" label="全部" />
          <Tab value="queued" label="排队中" />
          <Tab value="running" label="运行中" />
          <Tab value="paused" label="已暂停" />
          <Tab value="completed" label="已完成" />
          <Tab value="failed" label="失败" />
        </Tabs>
      </Box>

      <TableContainer component={Paper}>
        <Table aria-label="任务状态表格">
          <TableHead>
            <TableRow>
              <TableCell>任务名称</TableCell>
              <TableCell>目标模型</TableCell>
              <TableCell>攻击类型</TableCell>
              <TableCell>状态</TableCell>
              <TableCell>进度</TableCell>
              <TableCell>创建时间</TableCell>
              <TableCell align="right">操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <CircularProgress size={24} sx={{ my: 2 }} />
                  <Typography variant="body2">加载中...</Typography>
                </TableCell>
              </TableRow>
            ) : filteredTasks.length > 0 ? (
              filteredTasks.map((task) => (
                <TableRow key={task.id} hover>
                  <TableCell>{task.name}</TableCell>
                  <TableCell>{task.model}</TableCell>
                  <TableCell>{task.attack_type}</TableCell>
                  <TableCell>
                    <Chip 
                      label={statusMap[task.status]?.label || task.status} 
                      color={statusMap[task.status]?.color || "default"} 
                      size="small" 
                    />
                  </TableCell>
                  <TableCell>
                    {task.status === "queued" ? (
                      <Typography variant="body2">等待中</Typography>
                    ) : (
                      <Box sx={{ display: "flex", alignItems: "center" }}>
                        <Box sx={{ width: "100%", mr: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={task.progress || 0}
                          />
                        </Box>
                        <Box sx={{ minWidth: 35 }}>
                          <Typography variant="body2">
                            {task.progress || 0}%
                          </Typography>
                        </Box>
                      </Box>
                    )}
                  </TableCell>
                  <TableCell>{task.create_time}</TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
                      <Button
                        size="small"
                        onClick={() => handleViewDetails(task)}
                        sx={{ mr: 1 }}
                      >
                        详情
                      </Button>
                      {getStatusActions(task)}
                    </Box>
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  <Typography sx={{ py: 2 }}>
                    没有找到匹配的任务
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Box sx={{ mt: 3, display: "flex", justifyContent: "space-between" }}>
        <Button variant="contained" color="primary" onClick={handleCreateTask}>
          创建新任务
        </Button>
        <Button variant="outlined" onClick={handleRefresh} startIcon={loading ? <CircularProgress size={16} /> : null}>
          {loading ? "加载中..." : "刷新"}
        </Button>
      </Box>

      <TaskDetailDialog
        open={dialogOpen}
        task={selectedTask}
        onClose={handleCloseDialog}
        onTaskAction={handleTaskAction}
      />
    </Box>
  );
};

export default function Page() {
  return (
    <div>
      <h1>机器学习模型攻击系统</h1>
      <div style={{ marginTop: "30px", marginBottom: "30px" }}></div>
      <TaskStatus />
    </div>
  );
}
