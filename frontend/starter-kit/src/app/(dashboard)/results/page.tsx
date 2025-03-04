"use client";

import React, { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
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
import Slider from "@mui/material/Slider";
import FormControlLabel from "@mui/material/FormControlLabel";
import Switch from "@mui/material/Switch";

const API_URL = "http://127.0.0.1:5000";

// 攻击方法选项
const ATTACK_METHODS = [
  { id: "standard_attack", name: "基础逆向攻击" },
  { id: "PIG_attack", name: "PIG逆向攻击" },
  { id: "advanced", name: "高级逆向攻击" },
];

// 状态映射
const STATUS_MAP = {
  queued: { label: "排队中", color: "default" },
  running: { label: "运行中", color: "primary" },
  paused: { label: "已暂停", color: "warning" },
  completed: { label: "已完成", color: "success" },
  failed: { label: "失败", color: "error" }
};

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
      id={`attack-tabpanel-${index}`}
      aria-labelledby={`attack-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const AdvancedAttackResultsDisplay = () => {
  const [targetLabel, setTargetLabel] = useState<number>(0);
  const [attackMethod, setAttackMethod] = useState<string>("standard_attack");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [attackResult, setAttackResult] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [tabValue, setTabValue] = useState(0);
  const [attackHistory, setAttackHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  
  // 高级参数设置
  const [advancedSettings, setAdvancedSettings] = useState({
    iterations: 100,
    learningRate: 0.1,
    useRegularization: true,
  });

  // 加载历史攻击记录
  useEffect(() => {
    if (tabValue === 1) {
      fetchAttackHistory();
    }
  }, [tabValue]);

  // 获取历史攻击记录
  const fetchAttackHistory = async () => {
    try {
      setHistoryLoading(true);
      const response = await fetch(`${API_URL}/api/tasks?status=completed`);
      if (!response.ok) {
        throw new Error("获取历史记录失败");
      }
      const data = await response.json();
      setAttackHistory(data);
    } catch (error) {
      console.error("Error fetching history:", error);
    } finally {
      setHistoryLoading(false);
    }
  };

  // 更新高级参数
  const handleSettingChange = (setting: string, value: any) => {
    setAdvancedSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  // 处理标签页切换
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // 执行攻击
  const handleAttack = async () => {
    try {
      setIsLoading(true);
      setErrorMessage(null);
      setAttackResult(null);
      setPrediction(null);
      setConfidence(null);
      setCurrentTaskId(null);
      
      // 准备请求数据，包含高级参数
      const requestData = {
        target_label: targetLabel,
        attack_method: attackMethod,
        name: `${ATTACK_METHODS.find(m => m.id === attackMethod)?.name || "模型反演"} - 标签 ${targetLabel}`,
        model: "目标分类器",
        description: `针对标签 ${targetLabel} 的${ATTACK_METHODS.find(m => m.id === attackMethod)?.name || "模型反演"}攻击`,
        parameters: {
          iterations: advancedSettings.iterations,
          learning_rate: advancedSettings.learningRate,
          use_regularization: advancedSettings.useRegularization
        }
      };

      // 调用后端接口
      const response = await fetch(`${API_URL}/attack`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error(`攻击失败: ${response.statusText}`);
      }

      const data = await response.json();
      
      // 保存任务ID
      if (data.task_id) {
        setCurrentTaskId(data.task_id);
      }
      
      let imageUrl = "";
      // 如果后端返回base64图像
      if (data.image) {
        imageUrl = `data:image/png;base64,${data.image}`;
        setAttackResult(imageUrl);
      } 
      // 如果后端返回图像URL路径
      else if (data.result_image) {
        // 将相对路径转换为完整URL
        imageUrl = `${API_URL}${data.result_image.replace("./data", "/static")}`;
        setAttackResult(imageUrl);
      }

      // 获取模型预测结果
      const predictionData = await fetchPrediction();

    } catch (error) {
      console.error("Attack error:", error);
      setErrorMessage((error as Error).message || "执行攻击时发生错误");
    } finally {
      setIsLoading(false);
    }
  };

  // 获取预测结果
  const fetchPrediction = async () => {
    try {
      // 预测文件名是固定格式： inverted_{label}.png
      const imagePath = `/static/attack/inverted_${targetLabel}.png`;
      
      // 创建FormData对象
      const formData = new FormData();
      
      // 从URL获取图像并添加到FormData
      const imageResponse = await fetch(`${API_URL}${imagePath}`);
      const blob = await imageResponse.blob();
      formData.append("image_file", blob, `inverted_${targetLabel}.png`);
      
      // 发送预测请求
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("预测失败");
      }

      const predictionData = await response.json();
      
      let confidence = null;
      // 根据backend返回格式不同处理confidence
      if (predictionData.confidence) {
        confidence = predictionData.confidence;
      } else if (predictionData.confidences && Array.isArray(predictionData.confidences)) {
        // 获取最高置信度
        confidence = Math.max(...predictionData.confidences);
      }
      
      setPrediction(predictionData.prediction);
      setConfidence(confidence);
      
      // 如果存在任务ID，更新任务状态
      if (currentTaskId) {
        try {
          await fetch(`${API_URL}/api/tasks/${currentTaskId}/status`, {
            method: "PUT",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              status: "completed",
              progress: 100,
              result: {
                prediction: predictionData.prediction,
                confidence: confidence,
                success: predictionData.prediction === targetLabel
              }
            }),
          });
        } catch (err) {
          console.error("Error updating task status:", err);
        }
      }
      
      return {
        prediction: predictionData.prediction,
        confidence: confidence
      };
    } catch (error) {
      console.error("Prediction error:", error);
      
      // 如果预测失败，更新任务状态为失败
      if (currentTaskId) {
        try {
          await fetch(`${API_URL}/api/tasks/${currentTaskId}/status`, {
            method: "PUT",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              status: "failed",
              error_message: "预测失败: " + (error as Error).message
            }),
          });
        } catch (err) {
          console.error("Error updating task status:", err);
        }
      }
      
      return null;
    }
  };

  // 查看历史任务详情
  const viewTaskResult = async (taskId) => {
    try {
      const response = await fetch(`${API_URL}/api/tasks/${taskId}`);
      if (!response.ok) {
        throw new Error("获取任务详情失败");
      }
      
      const task = await response.json();
      
      // 设置当前目标标签
      if (task.target_label !== undefined) {
        setTargetLabel(task.target_label);
      }
      
      // 设置攻击方法
      if (task.attack_type) {
        const method = ATTACK_METHODS.find(m => m.name === task.attack_type);
        if (method) {
          setAttackMethod(method.id);
        }
      }
      
      // 加载图像和预测结果
      const imagePath = `/static/attack/inverted_${task.target_label}.png`;
      setAttackResult(`${API_URL}${imagePath}`);
      
      // 切换到攻击结果标签页
      setTabValue(0);
      
      // 尝试获取预测结果
      await fetchPrediction();
      
    } catch (error) {
      console.error("Error viewing task result:", error);
      setErrorMessage("加载任务结果失败");
    }
  };

  return (
    <Box
      sx={{
        padding: "20px",
        borderRadius: "8px",
        backgroundColor: "#f5f5f5",
        maxWidth: "900px",
        margin: "0 auto",
        boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
      }}
    >
      <Typography variant="h5" gutterBottom>
        高级攻击结果展示
      </Typography>
      
      <Tabs value={tabValue} onChange={handleTabChange} aria-label="attack tabs" sx={{ mb: 2 }}>
        <Tab label="执行攻击" />
        <Tab label="攻击历史" />
        <Tab label="高级设置" />
      </Tabs>
      
      <TabPanel value={tabValue} index={0}>
        <Typography variant="body1" gutterBottom>
          通过指定目标标签和攻击方法，执行模型逆向攻击并展示重建的隐私训练数据。
        </Typography>

        <Divider sx={{ margin: "20px 0" }} />

        <Grid container spacing={2}>
          {/* 目标标签输入 */}
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>目标标签：</Typography>
            <TextField
              type="number"
              value={targetLabel}
              onChange={(e) =>
                setTargetLabel(Math.max(0, parseInt(e.target.value) || 0))
              }
              fullWidth
              label="输入目标标签编号 (≥0)"
              variant="outlined"
            />
          </Grid>

          {/* 攻击方法选择 */}
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>攻击方法：</Typography>
            <FormControl fullWidth>
              <InputLabel id="attack-method-label">选择攻击方法</InputLabel>
              <Select
                labelId="attack-method-label"
                value={attackMethod}
                label="选择攻击方法"
                onChange={(e) => setAttackMethod(e.target.value)}
              >
                {ATTACK_METHODS.map((method) => (
                  <MenuItem key={method.id} value={method.id}>
                    {method.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>

        {/* 操作按钮 */}
        <Box sx={{ marginTop: "20px", marginBottom: "20px" }}>
          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={handleAttack}
            disabled={isLoading}
            startIcon={isLoading ? <CircularProgress size={20} /> : null}
          >
            {isLoading ? "正在执行攻击..." : "执行攻击"}
          </Button>
        </Box>

        {/* 错误信息 */}
        {errorMessage && (
          <Alert severity="error" sx={{ marginBottom: "20px" }}>
            {errorMessage}
          </Alert>
        )}

        {/* 攻击结果展示 */}
        {attackResult && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    重建的图像
                  </Typography>
                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "center",
                      marginTop: "10px",
                    }}
                  >
                    <img
                      src={attackResult}
                      alt={`重建的标签 ${targetLabel} 图像`}
                      style={{
                        maxWidth: "100%",
                        maxHeight: "250px",
                        objectFit: "contain",
                      }}
                    />
                  </Box>
                  {currentTaskId && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1, textAlign: 'right' }}>
                      任务ID: {currentTaskId}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    预测结果
                  </Typography>
                  {prediction !== null ? (
                    <Box>
                      <Typography variant="body1">
                        预测标签: <strong>{prediction}</strong>
                      </Typography>
                      {confidence !== null && (
                        <Typography variant="body1">
                          置信度: <strong>{(confidence * 100).toFixed(2)}%</strong>
                        </Typography>
                      )}
                      <Typography 
                        variant="body1" 
                        color={prediction === targetLabel ? "success.main" : "error.main"}
                        sx={{ marginTop: "10px" }}
                      >
                        {prediction === targetLabel 
                          ? "✓ 攻击成功! 预测结果与目标标签匹配" 
                          : "✗ 攻击失败! 预测结果与目标标签不匹配"}
                      </Typography>
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      暂无预测结果
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        <Typography variant="h6" gutterBottom>
          历史攻击记录
        </Typography>
        
        <Box sx={{ mb: 2, display: 'flex', justifyContent: 'flex-end' }}>
          <Button 
            variant="outlined" 
            onClick={fetchAttackHistory} 
            startIcon={historyLoading ? <CircularProgress size={16} /> : null}
          >
            {historyLoading ? "加载中..." : "刷新"}
          </Button>
        </Box>
        
        {historyLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        ) : attackHistory.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            暂无攻击历史记录
          </Typography>
        ) : (
          <Grid container spacing={2}>
            {attackHistory.map((task) => (
              <Grid item xs={12} md={6} lg={4} key={task.id}>
                <Card>
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      {task.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      目标标签: {task.target_label !== undefined ? task.target_label : '未知'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      攻击方法: {task.attack_type}
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                      <img
                        src={`${API_URL}/static/attack/inverted_${task.target_label}.png`}
                        alt={`历史攻击结果`}
                        style={{
                          maxWidth: "100%",
                          height: "120px",
                          objectFit: "contain",
                        }}
                        onError={(e) => {
                          e.currentTarget.src = "https://via.placeholder.com/150x150?text=无图像";
                        }}
                      />
                    </Box>
                    <Typography variant="caption" display="block" color="text.secondary">
                      创建时间: {task.create_time}
                    </Typography>
                    <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                      <Button 
                        size="small" 
                        variant="outlined" 
                        onClick={() => viewTaskResult(task.id)}
                      >
                        查看详情
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </TabPanel>
      
      <TabPanel value={tabValue} index={2}>
        <Typography variant="h6" gutterBottom>
          高级攻击参数
        </Typography>
        
        <Box sx={{ marginBottom: "20px" }}>
          <Typography gutterBottom>迭代次数:</Typography>
          <TextField
            type="number"
            fullWidth
            value={advancedSettings.iterations}
            onChange={(e) => handleSettingChange("iterations", Math.max(1, parseInt(e.target.value) || 1))}
            label="迭代次数 (≥1)"
            variant="outlined"
          />
          <Typography variant="caption" color="text.secondary">
            更高的迭代次数可能提供更好的攻击效果，但会增加攻击时间。
          </Typography>
        </Box>
        
        <Box sx={{ marginBottom: "20px" }}>
          <Typography gutterBottom>学习率: {advancedSettings.learningRate}</Typography>
          <Slider
            value={advancedSettings.learningRate}
            min={0.01}
            max={1}
            step={0.01}
            onChange={(_, value) => handleSettingChange("learningRate", value)}
            valueLabelDisplay="auto"
          />
          <Typography variant="caption" color="text.secondary">
            较高的学习率可能加快攻击速度，但可能降低重建图像质量。
          </Typography>
        </Box>
        
        <Box sx={{ marginBottom: "20px" }}>
          <FormControlLabel
            control={
              <Switch
                checked={advancedSettings.useRegularization}
                onChange={(e) => handleSettingChange("useRegularization", e.target.checked)}
              />
            }
            label="使用正则化"
          />
          <Typography variant="caption" color="text.secondary" display="block">
            开启正则化可以生成更清晰的图像，但可能降低攻击成功率。
          </Typography>
        </Box>
      </TabPanel>
    </Box>
  );
};

export default function AdvancedAttackResultsPage() {
  return (
    <div>
      <h1>机器学习模型攻击系统</h1>
      <div style={{ marginTop: "30px", marginBottom: "30px" }}></div>
      <AdvancedAttackResultsDisplay />
    </div>
  );
}