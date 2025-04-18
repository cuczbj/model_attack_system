"use client";

import React, { useState, useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Divider from "@mui/material/Divider";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Grid from "@mui/material/Grid";
import Alert from "@mui/material/Alert";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Paper from "@mui/material/Paper";
import Chip from "@mui/material/Chip";
import IconButton from "@mui/material/IconButton";
import RefreshIcon from "@mui/icons-material/Refresh";
import DownloadIcon from "@mui/icons-material/Download";
import CompareIcon from "@mui/icons-material/Compare";
import AssessmentIcon from "@mui/icons-material/Assessment";
import Tooltip from "@mui/material/Tooltip";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import Slider from "@mui/material/Slider";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import BarChart from "@mui/icons-material/BarChart";
import { styled } from "@mui/material/styles";

// API 基础URL
const API_URL = "http://127.0.0.1:5000";

// 自定义上传按钮样式
const VisuallyHiddenInput = styled("input")({
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  bottom: 0,
  left: 0,
  whiteSpace: "nowrap",
  width: 1,
});

// 标签面板组件
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`evaluation-tabpanel-${index}`}
      aria-labelledby={`evaluation-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// 任务类型接口
interface Task {
  id: string;
  name: string;
  model: string;
  attack_type: string;
  status: string;
  target_label: number;
  create_time: string;
  end_time?: string;
  image_path?: string;
}

// 评估结果接口
interface EvaluationResult {
  id: string;
  task_id: string;
  psnr: number;
  ssim: number;
  fid?: number;
  mse: number;
  target_accuracy: number;
  perceptual_similarity?: number;
  original_dataset?: string;
  create_time: string;
}

// 数据集接口
interface Dataset {
  id: string;
  name: string;
  path: string;
  description: string;
  class_count: number;
  image_count: number;
}

// 评估页面组件
export default function EvaluationPage() {
  // 状态管理
  const [tabValue, setTabValue] = useState(0);
  const [completedTasks, setCompletedTasks] = useState<Task[]>([]);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [evaluationResults, setEvaluationResults] = useState<EvaluationResult[]>([]);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [currentEvaluation, setCurrentEvaluation] = useState<EvaluationResult | null>(null);
  const [compareDialogOpen, setCompareDialogOpen] = useState(false);
  const [selectedOriginalImage, setSelectedOriginalImage] = useState<string | null>(null);
  const [uploadedDataset, setUploadedDataset] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [createEvaluationDialogOpen, setCreateEvaluationDialogOpen] = useState(false);
  const [batchEvaluationSettings, setBatchEvaluationSettings] = useState({
    evaluateAll: false,
    labelRange: { start: 0, end: 9 },
    attackMethods: ["standard_attack", "PIG_attack"],
    samplesPerLabel: 5,
    dataset: "",
  });
  const [batchEvaluationProgress, setBatchEvaluationProgress] = useState(0);
  const [batchEvaluationRunning, setBatchEvaluationRunning] = useState(false);
  const [batchEvaluationId, setBatchEvaluationId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // 加载已完成的攻击任务
  const loadCompletedTasks = async () => {
    try {
      setTasksLoading(true);
      setError(null);
      const response = await fetch(`${API_URL}/api/tasks?status=completed`);
      if (!response.ok) {
        throw new Error("获取已完成任务失败");
      }
      const data = await response.json();
      setCompletedTasks(data);
    } catch (error) {
      console.error("加载任务失败:", error);
      setError("无法加载已完成的攻击任务");
    } finally {
      setTasksLoading(false);
    }
  };

  // 加载评估结果
  const loadEvaluationResults = async () => {
    try {
      setResultsLoading(true);
      setError(null);
      const response = await fetch(`${API_URL}/api/evaluations/results`);
      if (!response.ok) {
        throw new Error("获取评估结果失败");
      }
      const data = await response.json();
      setEvaluationResults(data);
    } catch (error) {
      console.error("加载评估结果失败:", error);
      setError("无法加载评估结果");
    } finally {
      setResultsLoading(false);
    }
  };

  // 加载可用数据集
  const loadDatasets = async () => {
    try {
      setDatasetsLoading(true);
      setError(null);
      const response = await fetch(`${API_URL}/api/datasets`);
      if (!response.ok) {
        throw new Error("获取数据集失败");
      }
      const data = await response.json();
      setDatasets(data);
      if (data.length > 0 && !selectedDataset) {
        setSelectedDataset(data[0].id);
      }
    } catch (error) {
      console.error("加载数据集失败:", error);
      setError("无法加载可用数据集");
    } finally {
      setDatasetsLoading(false);
    }
  };

  // 评估单个任务
  const evaluateTask = async (taskId: string) => {
    try {
      setEvaluating(true);
      setError(null);

      // 获取选中的任务信息
      const task = completedTasks.find((t) => t.id === taskId);
      if (!task) {
        throw new Error("未找到选中的任务");
      }

      // 准备评估请求
      const requestData = {
        task_id: taskId,
        dataset_id: selectedDataset,
        metrics: ["psnr", "ssim", "fid", "mse"],
      };

      // 发送评估请求
      const response = await fetch(`${API_URL}/api/evaluations/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error("评估请求失败");
      }

      const result = await response.json();
      setCurrentEvaluation(result);

      // 刷新评估结果列表
      await loadEvaluationResults();
    } catch (error) {
      console.error("评估失败:", error);
      setError(`评估失败: ${error.message}`);
    } finally {
      setEvaluating(false);
    }
  };

  // 上传数据集
  const uploadDataset = async () => {
    if (!uploadedDataset) {
      setError("请先选择要上传的数据集文件");
      return;
    }

    try {
      setUploading(true);
      setUploadProgress(0);
      setError(null);

      const formData = new FormData();
      formData.append("dataset_file", uploadedDataset);
      formData.append("dataset_name", uploadedDataset.name.split(".")[0]);
      formData.append("description", "用户上传的数据集");

      // 模拟上传进度
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 5;
        });
      }, 300);

      const response = await fetch(`${API_URL}/api/datasets/upload`, {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error("上传数据集失败");
      }

      const result = await response.json();
      console.log("数据集上传成功:", result);

      // 重新加载数据集列表
      await loadDatasets();
      setUploadedDataset(null);

      // 选择新上传的数据集
      if (result.id) {
        setSelectedDataset(result.id);
      }
    } catch (error) {
      console.error("上传失败:", error);
      setError(`上传数据集失败: ${error.message}`);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  // 创建批量评估任务
  const createBatchEvaluation = async () => {
    try {
      setBatchEvaluationRunning(true);
      setError(null);

      const requestData = {
        dataset: batchEvaluationSettings.dataset || selectedDataset,
        attackMethods: batchEvaluationSettings.attackMethods,
        labelRange: batchEvaluationSettings.labelRange,
        samplesPerLabel: batchEvaluationSettings.samplesPerLabel,
        evaluateAll: batchEvaluationSettings.evaluateAll,
      };

      const response = await fetch(`${API_URL}/api/evaluations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error("创建批量评估任务失败");
      }

      const result = await response.json();
      console.log("批量评估任务已创建:", result);
      setBatchEvaluationId(result.id);

      // 关闭对话框
      setCreateEvaluationDialogOpen(false);

      // 轮询评估进度
      pollEvaluationProgress(result.id);
    } catch (error) {
      console.error("创建批量评估任务失败:", error);
      setError(`创建批量评估任务失败: ${error.message}`);
      setBatchEvaluationRunning(false);
    }
  };

  // 轮询批量评估进度
  const pollEvaluationProgress = async (evaluationId: string) => {
    try {
      const response = await fetch(`${API_URL}/api/evaluations/${evaluationId}`);
      if (!response.ok) {
        throw new Error("获取评估进度失败");
      }

      const data = await response.json();
      setBatchEvaluationProgress(data.progress);

      if (data.status === "completed" || data.status === "failed" || data.status === "stopped") {
        setBatchEvaluationRunning(false);
        
        if (data.status === "completed") {
          // 评估完成，刷新结果
          await loadEvaluationResults();
          setTabValue(1); // 切换到结果标签页
        } else if (data.status === "failed") {
          setError(`批量评估失败: ${data.error_message || "未知错误"}`);
        }

        return;
      }

      // 继续轮询
      setTimeout(() => pollEvaluationProgress(evaluationId), 2000);
    } catch (error) {
      console.error("轮询评估进度失败:", error);
      setError(`获取评估进度失败: ${error.message}`);
      setBatchEvaluationRunning(false);
    }
  };

  // 停止批量评估
  const stopBatchEvaluation = async () => {
    if (!batchEvaluationId) return;

    try {
      const response = await fetch(`${API_URL}/api/evaluations/${batchEvaluationId}/stop`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("停止评估任务失败");
      }

      console.log("评估任务已停止");
      setBatchEvaluationRunning(false);
    } catch (error) {
      console.error("停止评估任务失败:", error);
      setError(`停止评估任务失败: ${error.message}`);
    }
  };

  // 下载评估报告
  const downloadEvaluationReport = (evaluationId: string) => {
    window.open(`${API_URL}/api/evaluations/${evaluationId}/report`, "_blank");
  };

  // 打开图像比较对话框
  const openCompareDialog = (taskId: string) => {
    setSelectedTaskId(taskId);
    
    // 获取对应的原始图像（从选中的数据集和任务标签）
    const task = completedTasks.find((t) => t.id === taskId);
    const dataset = datasets.find((d) => d.id === selectedDataset);
    
    if (task && dataset) {
      const originalImagePath = `${API_URL}/api/datasets/${dataset.id}/images/${task.target_label}`;
      setSelectedOriginalImage(originalImagePath);
    }
    
    setCompareDialogOpen(true);
  };

  // 在组件挂载时加载数据
  useEffect(() => {
    loadCompletedTasks();
    loadEvaluationResults();
    loadDatasets();
  }, []);

  // 处理标签页切换
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // 处理数据集选择
  const handleDatasetChange = (event) => {
    setSelectedDataset(event.target.value);
  };

  // 获取评估结果表格行颜色
  const getRowColor = (value: number, metric: string) => {
    // 基于不同指标的良好值设置颜色
    if (metric === "psnr") {
      if (value > 30) return "success.light";
      if (value > 20) return "warning.light";
      return "error.light";
    } else if (metric === "ssim") {
      if (value > 0.8) return "success.light";
      if (value > 0.5) return "warning.light";
      return "error.light";
    } else if (metric === "fid") {
      if (value < 50) return "success.light";
      if (value < 100) return "warning.light";
      return "error.light";
    } else if (metric === "mse") {
      if (value < 0.01) return "success.light";
      if (value < 0.05) return "warning.light";
      return "error.light";
    } else if (metric === "target_accuracy") {
      if (value > 0.8) return "success.light";
      if (value > 0.5) return "warning.light";
      return "error.light";
    }
    return undefined;
  };

  return (
    <Box
      sx={{
        padding: "20px",
        borderRadius: "8px",
        backgroundColor: "#f5f5f5",
        maxWidth: "1200px",
        margin: "0 auto",
        boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
      }}
    >
      <Typography variant="h5" gutterBottom sx={{ display: "flex", alignItems: "center" }}>
        <AssessmentIcon sx={{ mr: 1 }} />
        攻击效果评估
      </Typography>
      
      <Tabs 
        value={tabValue} 
        onChange={handleTabChange} 
        aria-label="evaluation tabs" 
        sx={{ mb: 2, borderBottom: 1, borderColor: "divider" }}
      >
        <Tab label="单任务评估" />
        <Tab label="评估结果" />
        <Tab label="批量评估" />
        <Tab label="数据集管理" />
      </Tabs>
      
      {/* 错误提示 */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* 批量评估进度显示 */}
      {batchEvaluationRunning && (
        <Alert 
          severity="info" 
          sx={{ mb: 2 }}
          action={
            <Button color="inherit" size="small" onClick={stopBatchEvaluation}>
              停止
            </Button>
          }
        >
          批量评估进行中... {batchEvaluationProgress}%
          <Box sx={{ width: "100%", mt: 1 }}>
            <LinearProgress variant="determinate" value={batchEvaluationProgress} />
          </Box>
        </Alert>
      )}
      
      {/* 单任务评估标签页 */}
      <TabPanel value={tabValue} index={0}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="body1">
            选择已完成的攻击任务和原始数据集，评估攻击效果。评估指标包括PSNR、SSIM、FID等图像质量评价指标。
          </Typography>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
                  <Typography variant="h6">选择数据集</Typography>
                  <Button 
                    startIcon={<RefreshIcon />} 
                    size="small" 
                    onClick={loadDatasets}
                    disabled={datasetsLoading}
                  >
                    刷新
                  </Button>
                </Box>
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel id="dataset-select-label">原始数据集</InputLabel>
                  <Select
                    labelId="dataset-select-label"
                    value={selectedDataset}
                    label="原始数据集"
                    onChange={handleDatasetChange}
                    disabled={datasetsLoading || datasets.length === 0}
                  >
                    {datasetsLoading ? (
                      <MenuItem disabled value="">
                        加载中...
                      </MenuItem>
                    ) : datasets.length === 0 ? (
                      <MenuItem disabled value="">
                        无可用数据集
                      </MenuItem>
                    ) : (
                      datasets.map((dataset) => (
                        <MenuItem key={dataset.id} value={dataset.id}>
                          {dataset.name} ({dataset.class_count}类, {dataset.image_count}张图像)
                        </MenuItem>
                      ))
                    )}
                  </Select>
                </FormControl>
                
                {selectedDataset && datasets.find(d => d.id === selectedDataset) && (
                  <Box>
                    <Typography variant="subtitle2">数据集信息:</Typography>
                    <Typography variant="body2">
                      {datasets.find(d => d.id === selectedDataset)?.description || "无描述"}
                    </Typography>
                  </Box>
                )}
                
                {datasets.length === 0 && !datasetsLoading && (
                  <Alert severity="warning" sx={{ mt: 2 }}>
                    未找到可用数据集，请先上传数据集。
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
                  <Typography variant="h6">选择攻击任务</Typography>
                  <Button 
                    startIcon={<RefreshIcon />} 
                    size="small" 
                    onClick={loadCompletedTasks}
                    disabled={tasksLoading}
                  >
                    刷新
                  </Button>
                </Box>
                
                {tasksLoading ? (
                  <Box sx={{ display: "flex", justifyContent: "center", my: 2 }}>
                    <CircularProgress />
                  </Box>
                ) : completedTasks.length === 0 ? (
                  <Alert severity="info">
                    暂无已完成的攻击任务
                  </Alert>
                ) : (
                  <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
                    <Table stickyHeader size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>ID</TableCell>
                          <TableCell>目标标签</TableCell>
                          <TableCell>攻击方法</TableCell>
                          <TableCell>操作</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {completedTasks.map((task) => (
                          <TableRow 
                            key={task.id}
                            selected={selectedTaskId === task.id}
                            sx={{ cursor: "pointer" }}
                            onClick={() => setSelectedTaskId(task.id)}
                          >
                            <TableCell>{task.id}</TableCell>
                            <TableCell>{task.target_label}</TableCell>
                            <TableCell>{task.attack_type}</TableCell>
                            <TableCell>
                              <Tooltip title="评估">
                                <IconButton 
                                  size="small" 
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    evaluateTask(task.id);
                                  }}
                                  disabled={evaluating || !selectedDataset}
                                >
                                  <AssessmentIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="对比原始图像">
                                <IconButton 
                                  size="small"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    openCompareDialog(task.id);
                                  }}
                                  disabled={!selectedDataset}
                                >
                                  <CompareIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          {/* 已选择的任务展示 */}
          {selectedTaskId && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    选中的攻击任务
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={4}>
                      {(() => {
                        const task = completedTasks.find(t => t.id === selectedTaskId);
                        if (!task) return null;
                        
                        return (
                          <>
                            <Typography variant="subtitle1" gutterBottom>
                              任务信息
                            </Typography>
                            <Typography variant="body2">
                              <strong>ID:</strong> {task.id}
                            </Typography>
                            <Typography variant="body2">
                              <strong>名称:</strong> {task.name}
                            </Typography>
                            <Typography variant="body2">
                              <strong>目标标签:</strong> {task.target_label}
                            </Typography>
                            <Typography variant="body2">
                              <strong>攻击方法:</strong> {task.attack_type}
                            </Typography>
                            <Typography variant="body2">
                              <strong>完成时间:</strong> {task.end_time || "未知"}
                            </Typography>
                            
                            <Box sx={{ mt: 2 }}>
                              <Button
                                variant="contained"
                                color="primary"
                                onClick={() => evaluateTask(task.id)}
                                disabled={evaluating || !selectedDataset}
                                startIcon={evaluating ? <CircularProgress size={20} /> : <AssessmentIcon />}
                                fullWidth
                              >
                                {evaluating ? "评估中..." : "评估此任务"}
                              </Button>
                            </Box>
                          </>
                        );
                      })()}
                    </Grid>
                    
                    <Grid item xs={12} md={8}>
                      <Typography variant="subtitle1" gutterBottom>
                        攻击生成的图像
                      </Typography>
                      <Box 
                        sx={{ 
                          display: "flex", 
                          justifyContent: "center", 
                          border: "1px solid #ddd", 
                          borderRadius: 1,
                          padding: 2,
                          height: 250,
                          alignItems: "center"
                        }}
                      >
                        <img
                          src={`${API_URL}/api/tasks/${selectedTaskId}/image?t=${Date.now()}`}
                          alt="攻击生成图像"
                          style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                          onError={(e) => {
                            e.currentTarget.src = "https://via.placeholder.com/200x200?text=加载失败";
                          }}
                        />
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          )}
          
          {/* 评估结果展示 */}
          {currentEvaluation && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    评估结果
                  </Typography>
                  
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <TableContainer component={Paper}>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableCell>指标</TableCell>
                              <TableCell>值</TableCell>
                              <TableCell>评价</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            <TableRow sx={{ backgroundColor: getRowColor(currentEvaluation.psnr, "psnr") }}>
                              <TableCell>PSNR</TableCell>
                              <TableCell>{currentEvaluation.psnr.toFixed(2)} dB</TableCell>
                              <TableCell>
                                {currentEvaluation.psnr > 30 ? "优秀" : currentEvaluation.psnr > 20 ? "良好" : "较差"}
                              </TableCell>
                            </TableRow>
                            <TableRow sx={{ backgroundColor: getRowColor(currentEvaluation.ssim, "ssim") }}>
                              <TableCell>SSIM</TableCell>
                              <TableCell>{currentEvaluation.ssim.toFixed(4)}</TableCell>
                              <TableCell>
                                {currentEvaluation.ssim > 0.8 ? "优秀" : currentEvaluation.ssim > 0.5 ? "良好" : "较差"}
                              </TableCell>
                            </TableRow>
                            {currentEvaluation.fid !== undefined && (
                              <TableRow sx={{ backgroundColor: getRowColor(currentEvaluation.fid, "fid") }}>
                                <TableCell>FID</TableCell>
                                <TableCell>{currentEvaluation.fid.toFixed(2)}</TableCell>
                                <TableCell>
                                  {currentEvaluation.fid < 50 ? "优秀" : currentEvaluation.fid < 100 ? "良好" : "较差"}
                                </TableCell>
                              </TableRow>
                            )}
                            <TableRow sx={{ backgroundColor: getRowColor(currentEvaluation.mse, "mse") }}>
                              <TableCell>MSE</TableCell>
                              <TableCell>{currentEvaluation.mse.toFixed(4)}</TableCell>
                              <TableCell>
                                {currentEvaluation.mse < 0.01 ? "优秀" : currentEvaluation.mse < 0.05 ? "良好" : "较差"}
                              </TableCell>
                            </TableRow>
                            <TableRow sx={{ backgroundColor: getRowColor(currentEvaluation.target_accuracy, "target_accuracy") }}>
                              <TableCell>目标准确率</TableCell>
                              <TableCell>{(currentEvaluation.target_accuracy * 100).toFixed(2)}%</TableCell>
                              <TableCell>
                                {currentEvaluation.target_accuracy > 0.8 ? "优秀" : currentEvaluation.target_accuracy > 0.5 ? "良好" : "较差"}
                              </TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        评估分析
                      </Typography>
                      <Paper sx={{ p: 2, bgcolor: "background.default" }}>
                        <Typography variant="body1" gutterBottom>
                          综合评价: {
                            (currentEvaluation.psnr > 25 && currentEvaluation.ssim > 0.7) 
                              ? "攻击效果优秀，成功重建了原始隐私数据" 
                              : (currentEvaluation.psnr > 20 && currentEvaluation.ssim > 0.5)
                                ? "攻击效果良好，部分重建了原始隐私数据"
                                : "攻击效果一般，重建的图像与原始数据差异较大"
                          }
                        </Typography>
                        
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          • PSNR越高表示图像质量越好，通常大于30dB为优秀
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          • SSIM越接近1表示结构相似度越高，通常大于0.8为优秀
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          • FID越低表示生成图像分布越接近真实图像分布
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          • MSE越小表示误差越小，通常小于0.01为优秀
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          • 目标准确率表示模型对重建图像的识别准确度
                        </Typography>
                      </Paper>
                      
                      <Box sx={{ mt: 2 }}>
                        <Button
                          variant="outlined"
                          startIcon={<CompareIcon />}
                          onClick={() => openCompareDialog(selectedTaskId!)}
                          fullWidth
                        >
                          与原始图像对比
                        </Button>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </TabPanel>
      
      {/* 评估结果标签页 */}
      <TabPanel value={tabValue} index={1}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
          <Typography variant="h6">
            所有评估结果
          </Typography>
          
          <Button 
            variant="outlined" 
            startIcon={<RefreshIcon />}
            onClick={loadEvaluationResults}
            disabled={resultsLoading}
          >
            刷新结果
          </Button>
        </Box>
        
        {resultsLoading ? (
          <Box sx={{ display: "flex", justifyContent: "center", my: 4 }}>
            <CircularProgress />
          </Box>
        ) : evaluationResults.length === 0 ? (
          <Alert severity="info">
            暂无评估结果，请先进行任务评估。
          </Alert>
        ) : (
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>评估ID</TableCell>
                  <TableCell>任务ID</TableCell>
                  <TableCell>PSNR (dB)</TableCell>
                  <TableCell>SSIM</TableCell>
                  <TableCell>FID</TableCell>
                  <TableCell>MSE</TableCell>
                  <TableCell>目标准确率</TableCell>
                  <TableCell>创建时间</TableCell>
                  <TableCell>操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {evaluationResults.map((result) => (
                  <TableRow key={result.id}>
                    <TableCell>{result.id}</TableCell>
                    <TableCell>{result.task_id}</TableCell>
                    <TableCell sx={{ backgroundColor: getRowColor(result.psnr, "psnr") }}>
                      {result.psnr.toFixed(2)}
                    </TableCell>
                    <TableCell sx={{ backgroundColor: getRowColor(result.ssim, "ssim") }}>
                      {result.ssim.toFixed(4)}
                    </TableCell>
                    <TableCell sx={{ backgroundColor: result.fid ? getRowColor(result.fid, "fid") : undefined }}>
                      {result.fid ? result.fid.toFixed(2) : "N/A"}
                    </TableCell>
                    <TableCell sx={{ backgroundColor: getRowColor(result.mse, "mse") }}>
                      {result.mse.toFixed(4)}
                    </TableCell>
                    <TableCell sx={{ backgroundColor: getRowColor(result.target_accuracy, "target_accuracy") }}>
                      {(result.target_accuracy * 100).toFixed(2)}%
                    </TableCell>
                    <TableCell>{result.create_time}</TableCell>
                    <TableCell>
                      <Tooltip title="下载报告">
                        <IconButton onClick={() => downloadEvaluationReport(result.id)} size="small">
                          <DownloadIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="查看详情">
                        <IconButton 
                          size="small"
                          onClick={() => {
                            setSelectedTaskId(result.task_id);
                            setCurrentEvaluation(result);
                            setTabValue(0); // 切换到单任务评估页面
                          }}
                        >
                          <BarChart fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
        
        {/* 图表分析 */}
        {evaluationResults.length > 0 && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              评估结果分析
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" align="center" gutterBottom>
                      各攻击方法PSNR对比
                    </Typography>
                    <Box sx={{ height: 300, p: 2 }}>
                      {/* 图表将在后端实现 */}
                      <img
                        src={`${API_URL}/api/evaluations/charts/psnr_comparison?t=${Date.now()}`}
                        alt="PSNR对比图表"
                        style={{ width: "100%", height: "100%", objectFit: "contain" }}
                        onError={(e) => {
                          e.currentTarget.src = "https://via.placeholder.com/500x300?text=图表加载失败";
                        }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" align="center" gutterBottom>
                      各攻击方法SSIM对比
                    </Typography>
                    <Box sx={{ height: 300, p: 2 }}>
                      {/* 图表将在后端实现 */}
                      <img
                        src={`${API_URL}/api/evaluations/charts/ssim_comparison?t=${Date.now()}`}
                        alt="SSIM对比图表"
                        style={{ width: "100%", height: "100%", objectFit: "contain" }}
                        onError={(e) => {
                          e.currentTarget.src = "https://via.placeholder.com/500x300?text=图表加载失败";
                        }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}
      </TabPanel>
      
      {/* 批量评估标签页 */}
      <TabPanel value={tabValue} index={2}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="body1">
            批量评估多个标签和攻击方法，生成综合评估报告。
          </Typography>
        </Box>
        
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              创建批量评估任务
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel id="batch-dataset-label">选择数据集</InputLabel>
                  <Select
                    labelId="batch-dataset-label"
                    value={batchEvaluationSettings.dataset || selectedDataset}
                    label="选择数据集"
                    onChange={(e) => setBatchEvaluationSettings({
                      ...batchEvaluationSettings,
                      dataset: e.target.value
                    })}
                    disabled={datasetsLoading || datasets.length === 0 || batchEvaluationRunning}
                  >
                    {datasetsLoading ? (
                      <MenuItem disabled value="">加载中...</MenuItem>
                    ) : datasets.length === 0 ? (
                      <MenuItem disabled value="">无可用数据集</MenuItem>
                    ) : (
                      datasets.map((dataset) => (
                        <MenuItem key={dataset.id} value={dataset.id}>
                          {dataset.name}
                        </MenuItem>
                      ))
                    )}
                  </Select>
                </FormControl>
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={batchEvaluationSettings.evaluateAll}
                      onChange={(e) => setBatchEvaluationSettings({
                        ...batchEvaluationSettings,
                        evaluateAll: e.target.checked
                      })}
                      disabled={batchEvaluationRunning}
                    />
                  }
                  label="评估所有已完成任务"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Button 
                  variant="contained" 
                  color="primary" 
                  fullWidth
                  onClick={() => setCreateEvaluationDialogOpen(true)}
                  disabled={batchEvaluationRunning || datasetsLoading || datasets.length === 0}
                  sx={{ mt: 1 }}
                >
                  配置并创建批量评估任务
                </Button>
                
                {batchEvaluationRunning && (
                  <Button 
                    variant="outlined" 
                    color="error" 
                    fullWidth
                    onClick={stopBatchEvaluation}
                    sx={{ mt: 2 }}
                  >
                    停止当前评估任务
                  </Button>
                )}
              </Grid>
            </Grid>
          </CardContent>
        </Card>
        
        <Typography variant="h6" gutterBottom>
          批量评估历史
        </Typography>
        
        {/* 这里会显示批量评估历史 */}
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>名称</TableCell>
                <TableCell>数据集</TableCell>
                <TableCell>攻击方法</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>进度</TableCell>
                <TableCell>创建时间</TableCell>
                <TableCell>操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {/* 示例数据 */}
              <TableRow>
                <TableCell>eval-123456</TableCell>
                <TableCell>批量评估-1</TableCell>
                <TableCell>AT&T Faces</TableCell>
                <TableCell>标准, PIG</TableCell>
                <TableCell>
                  <Chip label="已完成" color="success" size="small" />
                </TableCell>
                <TableCell>100%</TableCell>
                <TableCell>2023-05-10 14:30:45</TableCell>
                <TableCell>
                  <Tooltip title="下载报告">
                    <IconButton size="small">
                      <DownloadIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>
      </TabPanel>
      
      {/* 数据集管理标签页 */}
      <TabPanel value={tabValue} index={3}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="body1">
            管理用于评估的原始数据集，支持上传新的数据集。
          </Typography>
        </Box>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={5}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  上传新数据集
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Button
                    component="label"
                    variant="contained"
                    startIcon={<CloudUploadIcon />}
                    fullWidth
                    disabled={uploading}
                  >
                    选择数据集文件
                    <VisuallyHiddenInput 
                      type="file" 
                      ref={fileInputRef}
                      onChange={(e) => {
                        if (e.target.files && e.target.files.length > 0) {
                          setUploadedDataset(e.target.files[0]);
                        }
                      }}
                      accept=".zip,.tar,.gz"
                    />
                  </Button>
                </Box>
                
                {uploadedDataset && (
                  <Typography variant="body2" gutterBottom>
                    已选择: {uploadedDataset.name} ({(uploadedDataset.size / (1024 * 1024)).toFixed(2)} MB)
                  </Typography>
                )}
                
                {uploadProgress > 0 && (
                  <Box sx={{ width: "100%", mb: 2 }}>
                    <LinearProgress variant="determinate" value={uploadProgress} />
                  </Box>
                )}
                
                <Button
                  variant="outlined"
                  color="primary"
                  fullWidth
                  onClick={uploadDataset}
                  disabled={!uploadedDataset || uploading}
                  startIcon={uploading ? <CircularProgress size={20} /> : null}
                >
                  {uploading ? "上传中..." : "上传数据集"}
                </Button>
                
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
                  支持的格式: ZIP, TAR, GZ 压缩文件，包含按类别组织的图像文件。
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={7}>
            <Card>
              <CardContent>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
                  <Typography variant="h6">可用数据集</Typography>
                  <Button 
                    startIcon={<RefreshIcon />} 
                    size="small" 
                    onClick={loadDatasets}
                    disabled={datasetsLoading}
                  >
                    刷新
                  </Button>
                </Box>
                
                {datasetsLoading ? (
                  <Box sx={{ display: "flex", justifyContent: "center", my: 2 }}>
                    <CircularProgress />
                  </Box>
                ) : datasets.length === 0 ? (
                  <Alert severity="info">
                    暂无可用数据集，请先上传数据集。
                  </Alert>
                ) : (
                  <TableContainer component={Paper}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>名称</TableCell>
                          <TableCell>类别数</TableCell>
                          <TableCell>图像数</TableCell>
                          <TableCell>路径</TableCell>
                          <TableCell>操作</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {datasets.map((dataset) => (
                          <TableRow key={dataset.id}>
                            <TableCell>{dataset.name}</TableCell>
                            <TableCell>{dataset.class_count}</TableCell>
                            <TableCell>{dataset.image_count}</TableCell>
                            <TableCell>{dataset.path}</TableCell>
                            <TableCell>
                              <Tooltip title="查看详情">
                                <IconButton size="small">
                                  <BarChart fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>
      
      {/* 图像比较对话框 */}
      <Dialog
        open={compareDialogOpen}
        onClose={() => setCompareDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>原始图像与攻击结果对比</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" align="center" gutterBottom>
                原始隐私图像
              </Typography>
              <Box 
                sx={{ 
                  display: "flex", 
                  justifyContent: "center", 
                  border: "1px solid #ddd", 
                  borderRadius: 1,
                  padding: 2,
                  height: 300,
                  alignItems: "center"
                }}
              >
                {selectedOriginalImage ? (
                  <img
                    src={selectedOriginalImage}
                    alt="原始图像"
                    style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                    onError={(e) => {
                      e.currentTarget.src = "https://via.placeholder.com/200x200?text=原始图像加载失败";
                    }}
                  />
                ) : (
                  <Typography color="text.secondary">未找到原始图像</Typography>
                )}
              </Box>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle1" align="center" gutterBottom>
                攻击重建图像
              </Typography>
              <Box 
                sx={{ 
                  display: "flex", 
                  justifyContent: "center", 
                  border: "1px solid #ddd", 
                  borderRadius: 1,
                  padding: 2,
                  height: 300,
                  alignItems: "center"
                }}
              >
                {selectedTaskId ? (
                  <img
                    src={`${API_URL}/api/tasks/${selectedTaskId}/image?t=${Date.now()}`}
                    alt="攻击结果图像"
                    style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                    onError={(e) => {
                      e.currentTarget.src = "https://via.placeholder.com/200x200?text=攻击图像加载失败";
                    }}
                  />
                ) : (
                  <Typography color="text.secondary">未选择攻击结果</Typography>
                )}
              </Box>
            </Grid>
            
            {currentEvaluation && (
              <Grid item xs={12}>
                <Typography variant="subtitle1" gutterBottom>
                  对比分析
                </Typography>
                <Paper sx={{ p: 2, bgcolor: "background.default" }}>
                  <Typography variant="body1" gutterBottom>
                    • PSNR: {currentEvaluation.psnr.toFixed(2)} dB ({currentEvaluation.psnr > 30 ? "优秀" : currentEvaluation.psnr > 20 ? "良好" : "较差"})
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    • SSIM: {currentEvaluation.ssim.toFixed(4)} ({currentEvaluation.ssim > 0.8 ? "优秀" : currentEvaluation.ssim > 0.5 ? "良好" : "较差"})
                  </Typography>
                  {currentEvaluation.fid !== undefined && (
                    <Typography variant="body1" gutterBottom>
                      • FID: {currentEvaluation.fid.toFixed(2)} ({currentEvaluation.fid < 50 ? "优秀" : currentEvaluation.fid < 100 ? "良好" : "较差"})
                    </Typography>
                  )}
                  <Typography variant="body1" gutterBottom>
                    • MSE: {currentEvaluation.mse.toFixed(4)} ({currentEvaluation.mse < 0.01 ? "优秀" : currentEvaluation.mse < 0.05 ? "良好" : "较差"})
                  </Typography>
                  <Typography variant="body1">
                    • 目标准确率: {(currentEvaluation.target_accuracy * 100).toFixed(2)}% ({currentEvaluation.target_accuracy > 0.8 ? "优秀" : currentEvaluation.target_accuracy > 0.5 ? "良好" : "较差"})
                  </Typography>
                </Paper>
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCompareDialogOpen(false)}>关闭</Button>
        </DialogActions>
      </Dialog>
      
      {/* 批量评估配置对话框 */}
      <Dialog
        open={createEvaluationDialogOpen}
        onClose={() => setCreateEvaluationDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>配置批量评估任务</DialogTitle>
        <DialogContent>
          <Typography variant="subtitle2" gutterBottom sx={{ mt: 1 }}>
            目标标签范围:
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                label="起始标签"
                type="number"
                value={batchEvaluationSettings.labelRange.start}
                onChange={(e) => setBatchEvaluationSettings({
                  ...batchEvaluationSettings,
                  labelRange: {
                    ...batchEvaluationSettings.labelRange,
                    start: parseInt(e.target.value) || 0
                  }
                })}
                fullWidth
                disabled={batchEvaluationSettings.evaluateAll}
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                label="结束标签"
                type="number"
                value={batchEvaluationSettings.labelRange.end}
                onChange={(e) => setBatchEvaluationSettings({
                  ...batchEvaluationSettings,
                  labelRange: {
                    ...batchEvaluationSettings.labelRange,
                    end: parseInt(e.target.value) || 0
                  }
                })}
                fullWidth
                disabled={batchEvaluationSettings.evaluateAll}
              />
            </Grid>
          </Grid>
          
          <Typography variant="subtitle2" gutterBottom sx={{ mt: 3 }}>
            每个标签的样本数:
          </Typography>
          <Slider
            value={batchEvaluationSettings.samplesPerLabel}
            min={1}
            max={20}
            step={1}
            marks={[
              { value: 1, label: '1' },
              { value: 5, label: '5' },
              { value: 10, label: '10' },
              { value: 20, label: '20' },
            ]}
            onChange={(_, value) => setBatchEvaluationSettings({
              ...batchEvaluationSettings,
              samplesPerLabel: value as number
            })}
            valueLabelDisplay="auto"
            disabled={batchEvaluationSettings.evaluateAll}
          />
          
          <Typography variant="subtitle2" gutterBottom sx={{ mt: 3 }}>
            选择攻击方法:
          </Typography>
          <Grid container spacing={1}>
            <Grid item>
              <Chip
                label="标准攻击"
                color={batchEvaluationSettings.attackMethods.includes("standard_attack") ? "primary" : "default"}
                onClick={() => {
                  const methods = [...batchEvaluationSettings.attackMethods];
                  const index = methods.indexOf("standard_attack");
                  if (index === -1) {
                    methods.push("standard_attack");
                  } else {
                    methods.splice(index, 1);
                  }
                  setBatchEvaluationSettings({
                    ...batchEvaluationSettings,
                    attackMethods: methods
                  });
                }}
                sx={{ m: 0.5 }}
              />
            </Grid>
            <Grid item>
              <Chip
                label="PIG攻击"
                color={batchEvaluationSettings.attackMethods.includes("PIG_attack") ? "primary" : "default"}
                onClick={() => {
                  const methods = [...batchEvaluationSettings.attackMethods];
                  const index = methods.indexOf("PIG_attack");
                  if (index === -1) {
                    methods.push("PIG_attack");
                  } else {
                    methods.splice(index, 1);
                  }
                  setBatchEvaluationSettings({
                    ...batchEvaluationSettings,
                    attackMethods: methods
                  });
                }}
                sx={{ m: 0.5 }}
              />
            </Grid>
            <Grid item>
              <Chip
                label="高级攻击"
                color={batchEvaluationSettings.attackMethods.includes("advanced") ? "primary" : "default"}
                onClick={() => {
                  const methods = [...batchEvaluationSettings.attackMethods];
                  const index = methods.indexOf("advanced");
                  if (index === -1) {
                    methods.push("advanced");
                  } else {
                    methods.splice(index, 1);
                  }
                  setBatchEvaluationSettings({
                    ...batchEvaluationSettings,
                    attackMethods: methods
                  });
                }}
                sx={{ m: 0.5 }}
              />
            </Grid>
          </Grid>
          
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            注意: 批量评估可能需要较长时间，请耐心等待。选择的标签范围和样本数越大，评估时间越长。
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateEvaluationDialogOpen(false)}>取消</Button>
          <Button 
            onClick={createBatchEvaluation} 
            variant="contained" 
            color="primary"
            disabled={
              batchEvaluationSettings.attackMethods.length === 0 ||
              (!batchEvaluationSettings.evaluateAll && 
               (batchEvaluationSettings.labelRange.start > batchEvaluationSettings.labelRange.end))
            }
          >
            创建评估任务
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
