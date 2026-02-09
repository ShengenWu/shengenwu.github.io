---
permalink: /
title: "Shengen WU - Homepage"  
excerpt: "M.Phil student at HKUST(GZ)"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
header:
  og_image: images/profile.png       # 【关键】告诉 Google 搜索和微信分享用这张图
  teaser: images/profile.png         # 站内缩略图
---

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>
My name is <span class="accent-text">Shengen WU</span>. I'm currently pursuing a Master of Philosophy (MPhil) degree at <a href="https://hkust-gz.edu.cn" class="link-accent">The Hong Kong University of Science and Technology (Guangzhou)</a>. I am a member of The Deep Interdisciplinary Intelligence Lab (Di² Lab), advised by Professor <a href="https://facultyprofiles.hkust-gz.edu.cn/faculty-personal-page/YUE-Yutao/yutaoyue" class="link-accent">Yuetao Yue</a>. My research focuses on <span class="accent-text">Large Language Models (LLMs)</span>, with particular interests in <span class="primary-gradient-text">multimodal learning</span> and <span class="primary-gradient-text">reasoning enhancement</span>. I received my Bachelor of Science degree in Financial Mathematics from <a href="https://xjtlu.edu.cn" class="link-accent">Xi'an Jiaotong-Liverpool University</a> and the <a href="https://liverpool.ac.uk" class="link-accent">University of Liverpool</a> in 2024.

<!-- <div class="quote-accent">
  I am passionate about:
    <ul>
      <li>Building <span class="primary-gradient-text">Generalist Agents</span>⚙️ that can perceive and act in the world.</li>
      <li>Enhancing <span class="primary-gradient-text">Reasoning Capabilities</span>🧠 of foundation models.</li>
    </ul>
</div> -->
<!-- 
Feel free to reach out if you'd like to discuss research or explore potential collaboration! -->

<!-- Removed Interdiscipline Researcher, Life Experiencer, World Explorer blocks to clean up layout -->

<span class='anchor' id='-news'></span>
# 🔥 News
- *01/2026*: &nbsp;🎉 One paper got accepted by <span class="accent-text">The Fourteenth International Conference on Learning Representations (ICLR 2026)</span>. See you in Rio de Janeiro🇧🇷!
- *09/2025*: &nbsp;🎉 One preprint was online via <span class="accent-text">ArXiv</span>.
- *09/2025*: &nbsp;🎉 One paper got accepted by <span class="accent-text">The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)</span>. See you in Suzhou🇨🇳!
- *01/2025*: &nbsp;🎉 One paper got accepted by <span class="accent-text">IEEE CSCWD 2025</span>.
- *06/2024*: &nbsp;🎉 One paper got accepted by <span class="accent-text">IEEE SEAI 2024</span>.

<span class='anchor' id='-educations'></span>
# 🏫 Educations
- *2024.09 - Present*: &nbsp;Master of Philosophy (MPhil) in Artificial Intelligence, <span class="primary-gradient-text">HKUST(GZ)</span><img src='images/hkustgzlogo.png' style="height:1em; vertical-align:middle;">.
- *2020.09 - 2024.06*: &nbsp;Bachelor of Science in Financial Mathematics, <span class="primary-gradient-text">Xi'an Jiaotong-Liverpool University</span><img src='images/xjtlu.png' style="height:1em; vertical-align:middle;"> & <span class="primary-gradient-text">University of Liverpool</span><img src='images/liverpool.png' style="height:1em; vertical-align:middle;">.

<span class='anchor' id='internships'></span>
# 💼 Internships
- *2025.08 - 2025.11*: &nbsp;Data Strategy Intern, <span class="primary-gradient-text">Douyin Group</span>, ByteDance.
- *2026.01 - Present*: &nbsp;Algorithm Engineer & Researcher, <span class="primary-gradient-text">HiThink Lab</span>, RoyalFlush.


# 📃 Publications

<div id="publications-wrapper">
  <div id="filter-container"></div>
  
  <div class='paper-box floating-card' data-tags="Knowledge Editing, LLM, ICLR">
    <div class='paper-box-image'>
      <div class="badge pulse-accent">ICLR 2026 Poster</div>
      <img src='images/paper-iclr2026.png' alt="ACE Overview" width="100%">
    </div>
    <div class='paper-box-text'>
      <h3>ACE: Attribution-Controlled Knowledge Editing for Multi-hop Factual Recall</h3>
      <div class="authors">Jiayu Yang, Yuxuan Fan, Songning Lai, <span class="primary-gradient-text">Shengen WU</span>, Jiaqi Tang, Chun Kang, Zhijiang Guo, Yutao Yue📧</div>
      <div class="venue">The Fourteenth International Conference on Learning Representations (ICLR 2026)</div>
      <div class="links">
        <a href="https://openreview.net/pdf?id=IuWIzmMvKo" class="btn-accent"><i class="fas fa-file-alt"></i> Paper</a>
      </div>
    </div>
  </div>
  
  <div class='paper-box floating-card' data-tags="Autonomous Driving, Survey, LLM">
    <div class='paper-box-image'>
      <div class="badge pulse-accent">ArXiv Preprint</div>
      <img src='images/paper-survey.png' alt="Survey Overview" width="100%">
    </div>
    <div class='paper-box-text'>
      <h3>Large Foundation Models for Trajectory Prediction in Autonomous Driving: A Comprehensive Survey</h3>
      <div class="authors">Wei Dai, <span class="primary-gradient-text">Shengen WU</span>, Wei Wu, Zhenhao Wang, Sisuo Lyu, Haicheng Liao, Limin Yu, Weiping Ding, Runwei Guan, Yutao Yue📧</div>
      <div class="venue">ArXiv Preprint (2025)</div>
      <div class="links">
        <a href="https://arxiv.org/abs/2509.10570" class="btn-accent"><i class="fas fa-file-alt"></i> Paper</a>
      </div>
    </div>
  </div>

  <div class='paper-box floating-card' data-tags="LLM, Multimodal, Table Understanding, EMNLP">
    <div class='paper-box-image'>
      <div class="badge pulse-accent">EMNLP 2025</div>
      <img src='images/paper-emnlp2025.png' alt="TableR1 Overview" width="100%">
    </div>
    <div class='paper-box-text'>
      <h3>Can GRPO Boost Complex Multimodal Table Understanding?</h3>
      <div class="authors">Xiaoqiang Kang, <span class="primary-gradient-text">Shengen WU</span>, Zimu Wang, Yilin Liu, Xiaobo Jin, Kaizhu Huang, Wei Wang, Yutao Yue, Xiaowei Huang, Qiufeng Wang📧</div>
      <div class="venue">The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)</div>
      <div class="links">
        <a href="https://aclanthology.org/2025.emnlp-main.637/" class="btn-accent"><i class="fas fa-file-alt"></i> Paper</a>
      </div>
    </div>
  </div>

  <div class='paper-box floating-card' data-tags="Medical Imaging, Zero-Shot Learning, Multimodal, CSCWD">
    <div class='paper-box-image'>
      <div class="badge pulse-accent">IEEE CSCWD 2025</div>
      <img src='images/paper-cscwd2025.png' alt="MMKNet Overview" width="100%">
    </div>
    <div class='paper-box-text'>
      <h3>MMKNet: A Multi-Modal Knowledge Network for Predicting Both Seen and Unseen Classes in Medical Imaging</h3>
      <div class="authors">Wenqi Xu, Hong-seng Gan📧, <span class="primary-gradient-text">Shengen WU</span>, Zimu Wang, Muhammad Hanif Ramlee, Wan Mahani Hafizah</div>
      <div class="venue">IEEE 29th International Conference on Computer Supported Cooperative Work and Design (CSCWD 2025)</div>
      <div class="links">
        <a href="https://ieeexplore.ieee.org/document/11033473" class="btn-accent"><i class="fas fa-file-alt"></i> Paper</a>
      </div>
    </div>
  </div>

  <div class='paper-box floating-card' data-tags="Medical Imaging, Zero-Shot Learning, Multimodal, SEAI">
    <div class='paper-box-image'>
      <div class="badge pulse-accent">IEEE SEAI 2024</div>
      <img src='images/paper-seai2024.png' alt="MVCNet Overview" width="100%">
    </div>
    <div class='paper-box-text'>
      <h3>MVCNet: A Vision Transformer-Based Network for Multi-Label Zero-Shot Learning in Medical Imaging</h3>
      <div class="authors"><span class="primary-gradient-text">Shengen Wu</span>, Hong-seng Gan📧, Ying-Tuan Lo, Muhammad Hanif Ramlee, Hafiz Basaruddin</div>
      <div class="venue">IEEE 4th International Conference on Software Engineering and Artificial Intelligence (SEAI 2024)</div>
      <div class="links">
        <a href="https://ieeexplore.ieee.org/abstract/document/10674183" class="btn-accent"><i class="fas fa-file-alt"></i> Paper</a>
      </div>
    </div>
  </div>
</div>

<!-- Awards, Talks, Services, Internships, Interests, Bond sections removed/cleaned. Please add back if needed. -->

<div style="text-align: center; margin-top: 2rem;">
  <a href="mailto:wushengen@outlook.com" class="btn-accent"><i class="fas fa-envelope"></i> Email Me</a>
</div>


<script>
document.addEventListener('DOMContentLoaded', function() {
  const wrapper = document.getElementById('publications-wrapper');
  if (!wrapper) return;

  const filterContainer = document.getElementById('filter-container');
  const paperBoxes = wrapper.querySelectorAll('.paper-box');
  
  let tagCounts = {}; 
  let activeTags = new Set();

  // 初始化：生成标签并统计数量
  paperBoxes.forEach(box => {
    const tagsAttribute = box.getAttribute('data-tags');
    if (tagsAttribute) {
      const tagsList = tagsAttribute.split(',').map(t => t.trim()).filter(t => t);
      
      // --- 插入标签到 Links 上方 ---
      const textContainer = box.querySelector('.paper-box-text');
      const linksContainer = box.querySelector('.links');
      
      if (textContainer && !textContainer.querySelector('.badge-container')) {
        const badgeContainer = document.createElement('div');
        badgeContainer.className = 'badge-container';
        
        tagsList.forEach(tag => {
          const badge = document.createElement('span');
          badge.className = 'inner-tag-badge';
          badge.textContent = tag;
          badgeContainer.appendChild(badge);
        });
        
        if (linksContainer) {
          textContainer.insertBefore(badgeContainer, linksContainer);
        } else {
          textContainer.appendChild(badgeContainer);
        }
      }
      // ---------------------------

      tagsList.forEach(tag => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });
    }
  });

  // 生成顶部过滤按钮
  const sortedTags = Object.keys(tagCounts).sort();
  if (filterContainer) {
    filterContainer.innerHTML = ''; 
    sortedTags.forEach(tag => {
      const btn = document.createElement('button');
      btn.className = 'filter-btn';
      btn.textContent = `${tag} (${tagCounts[tag]})`;
      
      btn.onclick = () => {
        if (activeTags.has(tag)) {
          activeTags.delete(tag);
          btn.classList.remove('active');
        } else {
          activeTags.add(tag);
          btn.classList.add('active');
        }
        filterPapers(); // 点击后触发过滤和高亮更新
      };
      
      filterContainer.appendChild(btn);
    });
  }

  // 🔥 核心逻辑更新：过滤论文 + 高亮标签
  function filterPapers() {
    paperBoxes.forEach(box => {
      // 1. 处理卡片显示/隐藏
      const boxTagsString = box.getAttribute('data-tags');
      const boxTags = boxTagsString ? boxTagsString.split(',').map(t => t.trim()) : [];
      
      let isVisible = true;
      if (activeTags.size > 0) {
        if (boxTags.length === 0) {
          isVisible = false;
        } else {
          // 必须包含所有选中的标签 (AND 逻辑)
          isVisible = Array.from(activeTags).every(activeTag => boxTags.includes(activeTag));
        }
      }

      if (isVisible) {
        box.classList.remove('hidden');
      } else {
        box.classList.add('hidden');
      }

      // 2. 🔥 处理内部标签的高亮 (即便卡片隐藏了，逻辑上也更新一下，没坏处)
      const innerBadges = box.querySelectorAll('.inner-tag-badge');
      innerBadges.forEach(badge => {
        // 如果这个小标签的文字，存在于 activeTags (顶部选中的集合) 中，就变色
        if (activeTags.has(badge.textContent)) {
          badge.classList.add('active');
        } else {
          badge.classList.remove('active');
        }
      });
    });
  }
});
</script>