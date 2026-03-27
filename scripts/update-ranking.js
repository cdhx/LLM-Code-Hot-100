/**
 * Auto-update README Top 20 ranking
 * Runs hourly via GitHub Actions
 */

const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');

// Complete topic ID mapping (matches index.html and vote.html)
const topicMeta = {
  // LLM Basics
  'gradient': { name: 'Gradient & Backprop', nameCN: '梯度与反向传播', category: 'Basics', link: 'docs/00-llm-basics.md#梯度与反向传播' },
  'linear-reg': { name: 'Linear Regression', nameCN: '线性回归', category: 'Basics', link: 'docs/00-llm-basics.md#线性回归' },
  'logistic-reg': { name: 'Logistic Regression', nameCN: '逻辑回归', category: 'Basics', link: 'docs/00-llm-basics.md#逻辑回归' },
  'softmax-reg': { name: 'Softmax Regression', nameCN: 'Softmax 回归', category: 'Basics', link: 'docs/00-llm-basics.md#softmax-回归' },
  'mlp': { name: 'MLP', nameCN: 'MLP 多层感知机', category: 'Basics', link: 'docs/00-llm-basics.md#mlp-多层感知机' },
  'activations': { name: 'Activations', nameCN: '激活函数', category: 'Basics', link: 'docs/00-llm-basics.md#激活函数' },
  
  // Attention
  'sdpa': { name: 'Scaled Dot-Product Attention', nameCN: 'Scaled Dot-Product Attention', category: 'Attention', link: 'docs/01-attention.md#scaled-dot-product-attention' },
  'mha': { name: 'Multi-Head Attention', nameCN: 'Multi-Head Attention', category: 'Attention', link: 'docs/01-attention.md#multi-head-attention' },
  'causal-mask': { name: 'Causal Mask', nameCN: 'Causal Mask', category: 'Attention', link: 'docs/01-attention.md#causal-mask' },
  'gqa': { name: 'GQA', nameCN: 'Grouped Query Attention', category: 'Attention', link: 'docs/01-attention.md#grouped-query-attention-gqa' },
  'mqa': { name: 'MQA', nameCN: 'Multi-Query Attention', category: 'Attention', link: 'docs/01-attention.md#multi-query-attention-mqa' },
  'flash-attn': { name: 'Flash Attention', nameCN: 'Flash Attention', category: 'Attention', link: 'docs/01-attention.md#flash-attention-原理' },
  'kv-cache-attn': { name: 'KV Cache', nameCN: 'KV Cache', category: 'Attention', link: 'docs/01-attention.md#kv-cache' },
  'cross-attn': { name: 'Cross Attention', nameCN: 'Cross Attention', category: 'Attention', link: 'docs/01-attention.md#cross-attention' },
  
  // Normalization
  'layernorm': { name: 'LayerNorm', nameCN: 'Layer Normalization', category: 'Norm', link: 'docs/02-normalization.md#layer-normalization' },
  'rmsnorm': { name: 'RMSNorm', nameCN: 'RMS Normalization', category: 'Norm', link: 'docs/02-normalization.md#rms-normalization' },
  'batchnorm': { name: 'BatchNorm', nameCN: 'Batch Normalization', category: 'Norm', link: 'docs/02-normalization.md#batch-normalization' },
  'prenorm': { name: 'Pre-Norm vs Post-Norm', nameCN: 'Pre-Norm vs Post-Norm', category: 'Norm', link: 'docs/02-normalization.md#pre-norm-vs-post-norm' },
  
  // Position Encoding
  'sinusoidal': { name: 'Sinusoidal PE', nameCN: 'Sinusoidal PE', category: 'Position', link: 'docs/03-position-encoding.md#sinusoidal-position-encoding' },
  'learnable-pe': { name: 'Learnable PE', nameCN: 'Learnable PE', category: 'Position', link: 'docs/03-position-encoding.md#learnable-position-encoding' },
  'rope': { name: 'RoPE', nameCN: 'RoPE 旋转位置编码', category: 'Position', link: 'docs/03-position-encoding.md#rotary-position-embedding-rope' },
  'alibi': { name: 'ALiBi', nameCN: 'ALiBi', category: 'Position', link: 'docs/03-position-encoding.md#alibi' },
  
  // Sampling
  'greedy': { name: 'Greedy Decoding', nameCN: 'Greedy Decoding', category: 'Sampling', link: 'docs/04-sampling.md#greedy-decoding' },
  'temperature': { name: 'Temperature', nameCN: 'Temperature Sampling', category: 'Sampling', link: 'docs/04-sampling.md#temperature-sampling' },
  'topk': { name: 'Top-k', nameCN: 'Top-k Sampling', category: 'Sampling', link: 'docs/04-sampling.md#top-k-sampling' },
  'topp': { name: 'Top-p', nameCN: 'Top-p Sampling', category: 'Sampling', link: 'docs/04-sampling.md#top-p-nucleus-sampling' },
  'beam': { name: 'Beam Search', nameCN: 'Beam Search', category: 'Sampling', link: 'docs/04-sampling.md#beam-search' },
  
  // Loss Functions
  'ce-loss': { name: 'Cross Entropy', nameCN: 'Cross Entropy Loss', category: 'Loss', link: 'docs/05-loss-functions.md#cross-entropy-loss' },
  'lm-loss': { name: 'LM Loss', nameCN: 'Language Model Loss', category: 'Loss', link: 'docs/05-loss-functions.md#language-model-loss' },
  'kl-div': { name: 'KL Divergence', nameCN: 'KL Divergence', category: 'Loss', link: 'docs/05-loss-functions.md#kl-divergence' },
  'mse-loss': { name: 'MSE Loss', nameCN: 'MSE Loss', category: 'Loss', link: 'docs/05-loss-functions.md#mse-loss' },
  'focal-loss': { name: 'Focal Loss', nameCN: 'Focal Loss', category: 'Loss', link: 'docs/05-loss-functions.md#focal-loss' },
  'sft-loss': { name: 'SFT Loss', nameCN: 'SFT Loss', category: 'Loss', link: 'docs/05-loss-functions.md#sft-loss' },
  'rm-loss': { name: 'Reward Model Loss', nameCN: 'Reward Model Loss', category: 'Loss', link: 'docs/05-loss-functions.md#reward-model-loss' },
  'contrastive': { name: 'Contrastive Loss', nameCN: 'Contrastive Loss', category: 'Loss', link: 'docs/05-loss-functions.md#contrastive-loss' },
  // Legacy IDs (for old votes)
  'ce': { name: 'Cross Entropy', nameCN: 'Cross Entropy Loss', category: 'Loss', link: 'docs/05-loss-functions.md#cross-entropy-loss' },
  'kl': { name: 'KL Divergence', nameCN: 'KL Divergence', category: 'Loss', link: 'docs/05-loss-functions.md#kl-divergence' },
  'mse': { name: 'MSE Loss', nameCN: 'MSE Loss', category: 'Loss', link: 'docs/05-loss-functions.md#mse-loss' },
  'focal': { name: 'Focal Loss', nameCN: 'Focal Loss', category: 'Loss', link: 'docs/05-loss-functions.md#focal-loss' },
  
  // Optimizers
  'sgd': { name: 'SGD', nameCN: 'SGD', category: 'Optimizer', link: 'docs/06-optimizers.md#sgd' },
  'momentum': { name: 'SGD+Momentum', nameCN: 'SGD + Momentum', category: 'Optimizer', link: 'docs/06-optimizers.md#sgd-with-momentum' },
  'adam': { name: 'Adam', nameCN: 'Adam', category: 'Optimizer', link: 'docs/06-optimizers.md#adam' },
  'adamw': { name: 'AdamW', nameCN: 'AdamW', category: 'Optimizer', link: 'docs/06-optimizers.md#adamw' },
  'lr-schedule': { name: 'LR Schedule', nameCN: '学习率调度', category: 'Optimizer', link: 'docs/06-optimizers.md#learning-rate-scheduler' },
  
  // Reinforcement Learning
  'reinforce': { name: 'REINFORCE', nameCN: 'REINFORCE', category: 'RL', link: 'docs/07-reinforcement-learning.md#reinforce' },
  'gae': { name: 'GAE', nameCN: 'GAE', category: 'RL', link: 'docs/07-reinforcement-learning.md#gae' },
  'ppo': { name: 'PPO', nameCN: 'PPO', category: 'RL', link: 'docs/07-reinforcement-learning.md#ppo' },
  'ppo-clip': { name: 'PPO-Clip', nameCN: 'PPO-Clip', category: 'RL', link: 'docs/07-reinforcement-learning.md#ppo' },
  'dpo': { name: 'DPO', nameCN: 'DPO', category: 'RL', link: 'docs/07-reinforcement-learning.md#dpo' },
  'grpo': { name: 'GRPO', nameCN: 'GRPO', category: 'RL', link: 'docs/07-reinforcement-learning.md#grpo' },
  'kl-penalty': { name: 'KL Penalty', nameCN: 'KL 惩罚', category: 'RL', link: 'docs/07-reinforcement-learning.md#ppo' },
  'reward-shaping': { name: 'Reward Shaping', nameCN: 'Reward Shaping', category: 'RL', link: 'docs/07-reinforcement-learning.md#ppo' },
  
  // Efficient Training
  'lora': { name: 'LoRA', nameCN: 'LoRA', category: 'Training', link: 'docs/08-efficient-training.md#lora' },
  'qlora': { name: 'QLoRA', nameCN: 'QLoRA', category: 'Training', link: 'docs/08-efficient-training.md#lora' },
  'grad-ckpt': { name: 'Gradient Checkpointing', nameCN: 'Gradient Checkpointing', category: 'Training', link: 'docs/08-efficient-training.md#gradient-checkpointing' },
  'mixed-precision': { name: 'Mixed Precision', nameCN: 'Mixed Precision', category: 'Training', link: 'docs/08-efficient-training.md#mixed-precision-training' },
  'grad-accum': { name: 'Gradient Accumulation', nameCN: 'Gradient Accumulation', category: 'Training', link: 'docs/08-efficient-training.md#gradient-accumulation' },
  
  // Inference Optimization
  'kv-cache': { name: 'KV Cache', nameCN: 'KV Cache', category: 'Inference', link: 'docs/09-inference-optimization.md#kv-cache' },
  'paged-attn': { name: 'Paged Attention', nameCN: 'Paged Attention', category: 'Inference', link: 'docs/09-inference-optimization.md#pagedattention' },
  'spec-decode': { name: 'Speculative Decoding', nameCN: 'Speculative Decoding', category: 'Inference', link: 'docs/09-inference-optimization.md#speculative-decoding' },
  'cont-batch': { name: 'Continuous Batching', nameCN: 'Continuous Batching', category: 'Inference', link: 'docs/09-inference-optimization.md#continuous-batching' },
  'quantization': { name: 'Quantization', nameCN: 'Quantization', category: 'Inference', link: 'docs/09-inference-optimization.md#kv-cache' },
  // Legacy
  'kv-cache-infer': { name: 'KV Cache', nameCN: 'KV Cache', category: 'Inference', link: 'docs/09-inference-optimization.md#kv-cache' },
  
  // Transformer Architecture
  'encoder-only': { name: 'Encoder-Only', nameCN: 'Encoder-Only (BERT)', category: 'Arch', link: 'docs/10-transformer-architecture.md#transformer-概览' },
  'decoder-only': { name: 'Decoder-Only', nameCN: 'Decoder-Only (GPT)', category: 'Arch', link: 'docs/10-transformer-architecture.md#gpt-style-decoder-only' },
  'enc-dec': { name: 'Encoder-Decoder', nameCN: 'Encoder-Decoder (T5)', category: 'Arch', link: 'docs/10-transformer-architecture.md#transformer-概览' },
  'ffn': { name: 'FFN', nameCN: 'FFN', category: 'Arch', link: 'docs/10-transformer-architecture.md#feed-forward-network' },
  'swiglu': { name: 'SwiGLU', nameCN: 'SwiGLU', category: 'Arch', link: 'docs/10-transformer-architecture.md#feed-forward-network' },
  
  // Legacy IDs for old votes
  'pre-post-norm': { name: 'Pre-Norm vs Post-Norm', nameCN: 'Pre-Norm vs Post-Norm', category: 'Norm', link: 'docs/02-normalization.md#pre-norm-vs-post-norm' },
};

async function main() {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_KEY;
  
  if (!supabaseUrl || !supabaseKey) {
    console.log('⚠️ Supabase not configured, skipping');
    return;
  }
  
  const supabase = createClient(supabaseUrl, supabaseKey);
  
  console.log('📊 Fetching votes...');
  const { data: votes, error } = await supabase.from('votes').select('topic_id');
  
  if (error) {
    console.error('❌ Failed:', error);
    return;
  }
  
  // Count votes
  const voteCounts = {};
  votes.forEach(v => {
    voteCounts[v.topic_id] = (voteCounts[v.topic_id] || 0) + 1;
  });
  
  // Sort and get Top 20
  const sorted = Object.entries(voteCounts)
    .map(([id, count]) => ({
      id, count,
      ...(topicMeta[id] || { name: id, nameCN: id, category: 'Other', link: '' })
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 20);
  
  console.log('🏆 Top 20:');
  sorted.forEach((t, i) => console.log(`  ${i + 1}. ${t.name} (${t.count} votes)`));
  
  const rankEmoji = ['🥇', '🥈', '🥉'];
  const today = new Date().toISOString().split('T')[0];
  
  // Generate English table
  const tableRowsEN = sorted.map((t, i) => {
    const rank = i < 3 ? rankEmoji[i] : (i + 1).toString();
    const nameLink = t.link ? `[${t.name}](${t.link})` : t.name;
    return `| ${rank} | ${nameLink} | ${t.category} | 🔥 ${t.count} |`;
  }).join('\n');
  
  const newTableEN = `## 🔥 Hot Top 20

> Community-driven, updated hourly via GitHub Actions
>
> **Last updated**: ${today}

| Rank | Topic | Category | Votes |
|:---:|:---|:---|:---:|
${tableRowsEN}`;

  // Generate Chinese table
  const tableRowsCN = sorted.map((t, i) => {
    const rank = i < 3 ? rankEmoji[i] : (i + 1).toString();
    const nameLink = t.link ? `[${t.nameCN}](${t.link})` : t.nameCN;
    return `| ${rank} | ${nameLink} | ${t.category} | 🔥 ${t.count} |`;
  }).join('\n');
  
  const newTableCN = `## 🔥 高频 Top 20

> 由社区投票驱动，每小时自动更新
>
> **最后更新**: ${today}

| 排名 | 题目 | 分类 | 票数 |
|:---:|:---|:---|:---:|
${tableRowsCN}`;

  // Update English README
  let contentEN = fs.readFileSync('README.md', 'utf-8');
  const regexEN = /## 🔥 Hot Top 20[\s\S]*?(?=\n---\n)/;
  if (regexEN.test(contentEN)) {
    contentEN = contentEN.replace(regexEN, newTableEN + '\n');
    fs.writeFileSync('README.md', contentEN);
    console.log('✅ README.md updated');
  }
  
  // Update Chinese README
  if (fs.existsSync('README_zh.md')) {
    let contentCN = fs.readFileSync('README_zh.md', 'utf-8');
    const regexCN = /## 🔥 高频 Top 20[\s\S]*?(?=\n---\n)/;
    if (regexCN.test(contentCN)) {
      contentCN = contentCN.replace(regexCN, newTableCN + '\n');
      fs.writeFileSync('README_zh.md', contentCN);
      console.log('✅ README_zh.md updated');
    }
  }
}

main().catch(console.error);
