/**
 * 自动更新 README 中的 Top 20 排行榜
 * 由 GitHub Actions 每天 00:00 运行
 */

const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// 题目映射表
const topicMeta = {
  // LLM 基础
  'gradient': { name: '梯度与反向传播', category: 'LLM基础' },
  'linear-reg': { name: '线性回归', category: 'LLM基础' },
  'logistic-reg': { name: '逻辑回归', category: 'LLM基础' },
  'softmax-reg': { name: 'Softmax 回归', category: 'LLM基础' },
  'mlp': { name: 'MLP 多层感知机', category: 'LLM基础' },
  'activations': { name: '激活函数', category: 'LLM基础' },
  
  // Attention
  'sdpa': { name: 'Scaled Dot-Product Attention', category: 'Attention' },
  'mha': { name: 'Multi-Head Attention', category: 'Attention' },
  'causal-mask': { name: 'Causal Mask', category: 'Attention' },
  'gqa': { name: 'Grouped Query Attention', category: 'Attention' },
  'mqa': { name: 'Multi-Query Attention', category: 'Attention' },
  'flash-attn': { name: 'Flash Attention', category: 'Attention' },
  'kv-cache': { name: 'KV Cache', category: 'Attention' },
  'cross-attn': { name: 'Cross Attention', category: 'Attention' },
  
  // 归一化
  'layernorm': { name: 'Layer Normalization', category: '归一化' },
  'rmsnorm': { name: 'RMS Normalization', category: '归一化' },
  'batchnorm': { name: 'Batch Normalization', category: '归一化' },
  'pre-post-norm': { name: 'Pre-Norm vs Post-Norm', category: '归一化' },
  
  // 位置编码
  'sinusoidal': { name: 'Sinusoidal PE', category: '位置编码' },
  'learnable-pe': { name: 'Learnable PE', category: '位置编码' },
  'rope': { name: 'RoPE 旋转位置编码', category: '位置编码' },
  'alibi': { name: 'ALiBi', category: '位置编码' },
  
  // 采样
  'greedy': { name: 'Greedy Decoding', category: '采样策略' },
  'temperature': { name: 'Temperature Sampling', category: '采样策略' },
  'topk': { name: 'Top-k Sampling', category: '采样策略' },
  'topp': { name: 'Top-p Sampling', category: '采样策略' },
  'beam': { name: 'Beam Search', category: '采样策略' },
  
  // 损失函数
  'ce': { name: 'Cross Entropy Loss', category: '损失函数' },
  'lm-loss': { name: 'Language Model Loss', category: '损失函数' },
  'kl': { name: 'KL Divergence', category: '损失函数' },
  'mse': { name: 'MSE Loss', category: '损失函数' },
  'focal': { name: 'Focal Loss', category: '损失函数' },
  'sft-loss': { name: 'SFT Loss', category: '损失函数' },
  'rm-loss': { name: 'Reward Model Loss', category: '损失函数' },
  'contrastive': { name: 'Contrastive Loss', category: '损失函数' },
  
  // 优化器
  'sgd': { name: 'SGD', category: '优化器' },
  'momentum': { name: 'SGD + Momentum', category: '优化器' },
  'adam': { name: 'Adam', category: '优化器' },
  'adamw': { name: 'AdamW', category: '优化器' },
  'lr-schedule': { name: '学习率调度', category: '优化器' },
  
  // 强化学习
  'reinforce': { name: 'REINFORCE', category: '强化学习' },
  'gae': { name: 'GAE', category: '强化学习' },
  'ppo': { name: 'PPO', category: '强化学习' },
  'ppo-clip': { name: 'PPO-Clip', category: '强化学习' },
  'dpo': { name: 'DPO', category: '强化学习' },
  'grpo': { name: 'GRPO', category: '强化学习' },
  'kl-penalty': { name: 'KL 惩罚', category: '强化学习' },
  'reward-shaping': { name: 'Reward Shaping', category: '强化学习' },
  
  // 高效训练
  'lora': { name: 'LoRA', category: '高效训练' },
  'qlora': { name: 'QLoRA', category: '高效训练' },
  'grad-ckpt': { name: 'Gradient Checkpointing', category: '高效训练' },
  'mixed-precision': { name: 'Mixed Precision', category: '高效训练' },
  'grad-accum': { name: 'Gradient Accumulation', category: '高效训练' },
  
  // 推理优化
  'kv-cache-infer': { name: 'KV Cache', category: '推理优化' },
  'paged-attn': { name: 'Paged Attention', category: '推理优化' },
  'spec-decode': { name: 'Speculative Decoding', category: '推理优化' },
  'cont-batch': { name: 'Continuous Batching', category: '推理优化' },
  'quantization': { name: 'Quantization', category: '推理优化' },
  
  // 架构
  'encoder-only': { name: 'Encoder-Only (BERT)', category: '架构' },
  'decoder-only': { name: 'Decoder-Only (GPT)', category: '架构' },
  'enc-dec': { name: 'Encoder-Decoder (T5)', category: '架构' },
  'ffn': { name: 'FFN', category: '架构' },
  'swiglu': { name: 'SwiGLU', category: '架构' },
};

async function main() {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    console.log('⚠️ Supabase 未配置，使用默认排名');
    return;
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey);
  
  // 获取投票数据
  console.log('📊 获取投票数据...');
  const { data: votes, error } = await supabase
    .from('votes')
    .select('topic_id');
  
  if (error) {
    console.error('❌ 获取数据失败:', error);
    return;
  }
  
  // 统计票数
  const voteCounts = {};
  votes.forEach(v => {
    voteCounts[v.topic_id] = (voteCounts[v.topic_id] || 0) + 1;
  });
  
  // 排序获取 Top 20
  const sorted = Object.entries(voteCounts)
    .map(([id, count]) => ({
      id,
      count,
      ...(topicMeta[id] || { name: id, category: '未知' })
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 20);
  
  console.log('🏆 Top 20:');
  sorted.forEach((t, i) => {
    console.log(`  ${i + 1}. ${t.name} (${t.count} votes)`);
  });
  
  // 生成表格
  const rankEmoji = ['🥇', '🥈', '🥉'];
  const tableRows = sorted.map((t, i) => {
    const rank = i < 3 ? rankEmoji[i] : (i + 1).toString();
    return `| ${rank} | ${t.name} | ${t.category} | 🔥 ${t.count} |`;
  }).join('\n');
  
  const newTable = `## 🔥 高频 Top 20

> 由社区投票实时产生，每日 00:00 自动更新
>
> **最后更新**: ${new Date().toISOString().split('T')[0]}

| 排名 | 题目 | 分类 | 票数 |
|:---:|:---|:---|:---:|
${tableRows}`;

  // 读取 README
  const readmePath = path.join(__dirname, '..', 'README.md');
  let readme = fs.readFileSync(readmePath, 'utf-8');
  
  // 替换 Top 20 部分
  const top20Regex = /## 🔥 高频 Top 20[\s\S]*?(?=\n---\n)/;
  
  if (top20Regex.test(readme)) {
    readme = readme.replace(top20Regex, newTable + '\n');
    fs.writeFileSync(readmePath, readme);
    console.log('✅ README 已更新');
  } else {
    console.log('⚠️ 未找到 Top 20 部分，跳过更新');
  }
}

main().catch(console.error);
