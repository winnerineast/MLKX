



<!DOCTYPE html>
<html lang="en" class="">
  <head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# object: http://ogp.me/ns/object# article: http://ogp.me/ns/article# profile: http://ogp.me/ns/profile#">
    <meta charset='utf-8'>
    

    <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/frameworks-7f2a605da6435efc2ee84e59714b38adf17b327656cf5739a7f65a0029fbe2d5.css" media="all" rel="stylesheet" />
    <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/github-1739262060807fd695418fdde2e31ce8d76b981c889a19ef01f1c25dd1b378df.css" media="all" rel="stylesheet" />
    
    
    <link crossorigin="anonymous" href="https://assets-cdn.github.com/assets/site-0aa4f1a4bd9907ebd09f265a95d4856cabf513dd39b891f737b1b35c39f12564.css" media="all" rel="stylesheet" />
    

    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta http-equiv="Content-Language" content="en">
    <meta name="viewport" content="width=device-width">
    
    <title>MachineLearning/neuralNets.md at master · andrewt3000/MachineLearning · GitHub</title>
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
    <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
    <link rel="apple-touch-icon" href="/apple-touch-icon.png">
    <link rel="apple-touch-icon" sizes="57x57" href="/apple-touch-icon-57x57.png">
    <link rel="apple-touch-icon" sizes="60x60" href="/apple-touch-icon-60x60.png">
    <link rel="apple-touch-icon" sizes="72x72" href="/apple-touch-icon-72x72.png">
    <link rel="apple-touch-icon" sizes="76x76" href="/apple-touch-icon-76x76.png">
    <link rel="apple-touch-icon" sizes="114x114" href="/apple-touch-icon-114x114.png">
    <link rel="apple-touch-icon" sizes="120x120" href="/apple-touch-icon-120x120.png">
    <link rel="apple-touch-icon" sizes="144x144" href="/apple-touch-icon-144x144.png">
    <link rel="apple-touch-icon" sizes="152x152" href="/apple-touch-icon-152x152.png">
    <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon-180x180.png">
    <meta property="fb:app_id" content="1401488693436528">

      <meta content="https://avatars1.githubusercontent.com/u/838741?v=3&amp;s=400" name="twitter:image:src" /><meta content="@github" name="twitter:site" /><meta content="summary" name="twitter:card" /><meta content="andrewt3000/MachineLearning" name="twitter:title" /><meta content="MachineLearning - Machine Learning Notes" name="twitter:description" />
      <meta content="https://avatars1.githubusercontent.com/u/838741?v=3&amp;s=400" property="og:image" /><meta content="GitHub" property="og:site_name" /><meta content="object" property="og:type" /><meta content="andrewt3000/MachineLearning" property="og:title" /><meta content="https://github.com/andrewt3000/MachineLearning" property="og:url" /><meta content="MachineLearning - Machine Learning Notes" property="og:description" />
      <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">
    <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">
    <link rel="assets" href="https://assets-cdn.github.com/">
    
    <meta name="pjax-timeout" content="1000">
    
    <meta name="request-id" content="7342E3FB:2C138:7C13C8:584C294B" data-pjax-transient>

    <meta name="msapplication-TileImage" content="/windows-tile.png">
    <meta name="msapplication-TileColor" content="#ffffff">
    <meta name="selected-link" value="repo_source" data-pjax-transient>

    <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
<meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
    <meta name="google-analytics" content="UA-3769691-2">

<meta content="collector.githubapp.com" name="octolytics-host" /><meta content="github" name="octolytics-app-id" /><meta content="7342E3FB:2C138:7C13C8:584C294B" name="octolytics-dimension-request_id" />
<meta content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" name="analytics-location" />



  <meta class="js-ga-set" name="dimension1" content="Logged Out">



        <meta name="hostname" content="github.com">
    <meta name="user-login" content="">

        <meta name="expected-hostname" content="github.com">
      <meta name="js-proxy-site-detection-payload" content="Nzc4YWE3NDljMDg5YjExMjdjYmZjNmQwODM2YTQ5YmNkYTczYzUwNWNhYWYzZTRmOTZjZTBmNTZkMzVjYzI2Znx7InJlbW90ZV9hZGRyZXNzIjoiMTE1LjY2LjIyNy4yNTEiLCJyZXF1ZXN0X2lkIjoiNzM0MkUzRkI6MkMxMzg6N0MxM0M4OjU4NEMyOTRCIiwidGltZXN0YW1wIjoxNDgxMzg2MzE2LCJob3N0IjoiZ2l0aHViLmNvbSJ9">


      <link rel="mask-icon" href="https://assets-cdn.github.com/pinned-octocat.svg" color="#000000">
      <link rel="icon" type="image/x-icon" href="https://assets-cdn.github.com/favicon.ico">

    <meta name="html-safe-nonce" content="58c8aab826ab7d1768a2245e1dcfa7153aea9b5c">
    <meta content="3a022934272296fceb2761208918067b8eb2331c" name="form-nonce" />

    <meta http-equiv="x-pjax-version" content="7144cf2e659ee16f8240cf9ac26214f8">
    

      
  <meta name="description" content="MachineLearning - Machine Learning Notes">
  <meta name="go-import" content="github.com/andrewt3000/MachineLearning git https://github.com/andrewt3000/MachineLearning.git">

  <meta content="838741" name="octolytics-dimension-user_id" /><meta content="andrewt3000" name="octolytics-dimension-user_login" /><meta content="64251635" name="octolytics-dimension-repository_id" /><meta content="andrewt3000/MachineLearning" name="octolytics-dimension-repository_nwo" /><meta content="true" name="octolytics-dimension-repository_public" /><meta content="false" name="octolytics-dimension-repository_is_fork" /><meta content="64251635" name="octolytics-dimension-repository_network_root_id" /><meta content="andrewt3000/MachineLearning" name="octolytics-dimension-repository_network_root_nwo" />
  <link href="https://github.com/andrewt3000/MachineLearning/commits/master.atom" rel="alternate" title="Recent Commits to MachineLearning:master" type="application/atom+xml">


      <link rel="canonical" href="https://github.com/andrewt3000/MachineLearning/blob/master/neuralNets.md" data-pjax-transient>
  </head>


  <body class="logged-out  env-production  vis-public page-blob">
    <div id="js-pjax-loader-bar" class="pjax-loader-bar"><div class="progress"></div></div>
    <a href="#start-of-content" tabindex="1" class="accessibility-aid js-skip-to-content">Skip to content</a>

    
    
    



          <header class="site-header js-details-container" role="banner">
  <div class="container-responsive">
    <a class="header-logo-invertocat" href="https://github.com/" aria-label="Homepage" data-ga-click="(Logged out) Header, go to homepage, icon:logo-wordmark">
      <svg aria-hidden="true" class="octicon octicon-mark-github" height="32" version="1.1" viewBox="0 0 16 16" width="32"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
    </a>

    <button class="btn-link float-right site-header-toggle js-details-target" type="button" aria-label="Toggle navigation">
      <svg aria-hidden="true" class="octicon octicon-three-bars" height="24" version="1.1" viewBox="0 0 12 16" width="18"><path fill-rule="evenodd" d="M11.41 9H.59C0 9 0 8.59 0 8c0-.59 0-1 .59-1H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1h.01zm0-4H.59C0 5 0 4.59 0 4c0-.59 0-1 .59-1H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1h.01zM.59 11H11.4c.59 0 .59.41.59 1 0 .59 0 1-.59 1H.59C0 13 0 12.59 0 12c0-.59 0-1 .59-1z"/></svg>
    </button>

    <div class="site-header-menu">
      <nav class="site-header-nav site-header-nav-main">
        <a href="/personal" class="js-selected-navigation-item nav-item nav-item-personal" data-ga-click="Header, click, Nav menu - item:personal" data-selected-links="/personal /personal">
          Personal
</a>        <a href="/open-source" class="js-selected-navigation-item nav-item nav-item-opensource" data-ga-click="Header, click, Nav menu - item:opensource" data-selected-links="/open-source /open-source">
          Open source
</a>        <a href="/business" class="js-selected-navigation-item nav-item nav-item-business" data-ga-click="Header, click, Nav menu - item:business" data-selected-links="/business /business/partners /business/features /business/customers /business">
          Business
</a>        <a href="/explore" class="js-selected-navigation-item nav-item nav-item-explore" data-ga-click="Header, click, Nav menu - item:explore" data-selected-links="/explore /trending /trending/developers /integrations /integrations/feature/code /integrations/feature/collaborate /integrations/feature/ship /showcases /explore">
          Explore
</a>      </nav>

      <div class="site-header-actions">
            <a class="btn btn-primary site-header-actions-btn" href="/join?source=header-repo" data-ga-click="(Logged out) Header, clicked Sign up, text:sign-up">Sign up</a>
          <a class="btn site-header-actions-btn mr-1" href="/login?return_to=%2Fandrewt3000%2FMachineLearning%2Fblob%2Fmaster%2FneuralNets.md" data-ga-click="(Logged out) Header, clicked Sign in, text:sign-in">Sign in</a>
      </div>

        <nav class="site-header-nav site-header-nav-secondary mr-md-3">
          <a class="nav-item" href="/pricing">Pricing</a>
          <a class="nav-item" href="/blog">Blog</a>
          <a class="nav-item" href="https://help.github.com">Support</a>
          <a class="nav-item header-search-link" href="https://github.com/search">Search GitHub</a>
              <div class="header-search scoped-search site-scoped-search js-site-search" role="search">
  <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="/andrewt3000/MachineLearning/search" class="js-site-search-form" data-scoped-search-url="/andrewt3000/MachineLearning/search" data-unscoped-search-url="/search" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
    <label class="form-control header-search-wrapper js-chromeless-input-container">
      <div class="header-search-scope">This repository</div>
      <input type="text"
        class="form-control header-search-input js-site-search-focus js-site-search-field is-clearable"
        data-hotkey="s"
        name="q"
        placeholder="Search"
        aria-label="Search this repository"
        data-unscoped-placeholder="Search GitHub"
        data-scoped-placeholder="Search"
        autocapitalize="off">
    </label>
</form></div>

        </nav>
    </div>
  </div>
</header>



    <div id="start-of-content" class="accessibility-aid"></div>

      <div id="js-flash-container">
</div>


    <div role="main">
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode">
    <div id="js-repo-pjax-container" data-pjax-container>
      
<div class="pagehead repohead instapaper_ignore readability-menu experiment-repo-nav">
  <div class="container repohead-details-container">

    

<ul class="pagehead-actions">

  <li>
      <a href="/login?return_to=%2Fandrewt3000%2FMachineLearning"
    class="btn btn-sm btn-with-count tooltipped tooltipped-n"
    aria-label="You must be signed in to watch a repository" rel="nofollow">
    <svg aria-hidden="true" class="octicon octicon-eye" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
    Watch
  </a>
  <a class="social-count" href="/andrewt3000/MachineLearning/watchers"
     aria-label="1 user is watching this repository">
    1
  </a>

  </li>

  <li>
      <a href="/login?return_to=%2Fandrewt3000%2FMachineLearning"
    class="btn btn-sm btn-with-count tooltipped tooltipped-n"
    aria-label="You must be signed in to star a repository" rel="nofollow">
    <svg aria-hidden="true" class="octicon octicon-star" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74z"/></svg>
    Star
  </a>

    <a class="social-count js-social-count" href="/andrewt3000/MachineLearning/stargazers"
      aria-label="10 users starred this repository">
      10
    </a>

  </li>

  <li>
      <a href="/login?return_to=%2Fandrewt3000%2FMachineLearning"
        class="btn btn-sm btn-with-count tooltipped tooltipped-n"
        aria-label="You must be signed in to fork a repository" rel="nofollow">
        <svg aria-hidden="true" class="octicon octicon-repo-forked" height="16" version="1.1" viewBox="0 0 10 16" width="10"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
        Fork
      </a>

    <a href="/andrewt3000/MachineLearning/network" class="social-count"
       aria-label="6 users forked this repository">
      6
    </a>
  </li>
</ul>

    <h1 class="public ">
  <svg aria-hidden="true" class="octicon octicon-repo" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
  <span class="author" itemprop="author"><a href="/andrewt3000" class="url fn" rel="author">andrewt3000</a></span><!--
--><span class="path-divider">/</span><!--
--><strong itemprop="name"><a href="/andrewt3000/MachineLearning" data-pjax="#js-repo-pjax-container">MachineLearning</a></strong>

</h1>

  </div>
  <div class="container">
    
<nav class="reponav js-repo-nav js-sidenav-container-pjax"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
     role="navigation"
     data-pjax="#js-repo-pjax-container">

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a href="/andrewt3000/MachineLearning" class="js-selected-navigation-item selected reponav-item" data-hotkey="g c" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches /andrewt3000/MachineLearning" itemprop="url">
      <svg aria-hidden="true" class="octicon octicon-code" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg>
      <span itemprop="name">Code</span>
      <meta itemprop="position" content="1">
</a>  </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a href="/andrewt3000/MachineLearning/issues" class="js-selected-navigation-item reponav-item" data-hotkey="g i" data-selected-links="repo_issues repo_labels repo_milestones /andrewt3000/MachineLearning/issues" itemprop="url">
        <svg aria-hidden="true" class="octicon octicon-issue-opened" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"/></svg>
        <span itemprop="name">Issues</span>
        <span class="counter">0</span>
        <meta itemprop="position" content="2">
</a>    </span>

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a href="/andrewt3000/MachineLearning/pulls" class="js-selected-navigation-item reponav-item" data-hotkey="g p" data-selected-links="repo_pulls /andrewt3000/MachineLearning/pulls" itemprop="url">
      <svg aria-hidden="true" class="octicon octicon-git-pull-request" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M11 11.28V5c-.03-.78-.34-1.47-.94-2.06C9.46 2.35 8.78 2.03 8 2H7V0L4 3l3 3V4h1c.27.02.48.11.69.31.21.2.3.42.31.69v6.28A1.993 1.993 0 0 0 10 15a1.993 1.993 0 0 0 1-3.72zm-1 2.92c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zM4 3c0-1.11-.89-2-2-2a1.993 1.993 0 0 0-1 3.72v6.56A1.993 1.993 0 0 0 2 15a1.993 1.993 0 0 0 1-3.72V4.72c.59-.34 1-.98 1-1.72zm-.8 10c0 .66-.55 1.2-1.2 1.2-.65 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
      <span itemprop="name">Pull requests</span>
      <span class="counter">0</span>
      <meta itemprop="position" content="3">
</a>  </span>

  <a href="/andrewt3000/MachineLearning/projects" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /andrewt3000/MachineLearning/projects">
    <svg aria-hidden="true" class="octicon octicon-project" height="16" version="1.1" viewBox="0 0 15 16" width="15"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
    Projects
    <span class="counter">0</span>
</a>


  <a href="/andrewt3000/MachineLearning/pulse" class="js-selected-navigation-item reponav-item" data-selected-links="pulse /andrewt3000/MachineLearning/pulse">
    <svg aria-hidden="true" class="octicon octicon-pulse" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M11.5 8L8.8 5.4 6.6 8.5 5.5 1.6 2.38 8H0v2h3.6l.9-1.8.9 5.4L9 8.5l1.6 1.5H14V8z"/></svg>
    Pulse
</a>
  <a href="/andrewt3000/MachineLearning/graphs" class="js-selected-navigation-item reponav-item" data-selected-links="repo_graphs repo_contributors /andrewt3000/MachineLearning/graphs">
    <svg aria-hidden="true" class="octicon octicon-graph" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M16 14v1H0V0h1v14h15zM5 13H3V8h2v5zm4 0H7V3h2v10zm4 0h-2V6h2v7z"/></svg>
    Graphs
</a>

</nav>

  </div>
</div>

<div class="container new-discussion-timeline experiment-repo-nav">
  <div class="repository-content">

    

<a href="/andrewt3000/MachineLearning/blob/bc9b2767ef0d5c2da6b170da52b8539b1d3ea0c0/neuralNets.md" class="d-none js-permalink-shortcut" data-hotkey="y">Permalink</a>

<!-- blob contrib key: blob_contributors:v21:17b6ec7d1eaed82f0f0365f1a3491ab0 -->

<div class="file-navigation js-zeroclipboard-container">
  
<div class="select-menu branch-select-menu js-menu-container js-select-menu float-left">
  <button class="btn btn-sm select-menu-button js-menu-target css-truncate" data-hotkey="w"
    
    type="button" aria-label="Switch branches or tags" tabindex="0" aria-haspopup="true">
    <i>Branch:</i>
    <span class="js-select-button css-truncate-target">master</span>
  </button>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax aria-hidden="true">

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <svg aria-label="Close" class="octicon octicon-x js-menu-close" height="16" role="img" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
        <span class="select-menu-title">Switch branches/tags</span>
      </div>

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Filter branches/tags" id="context-commitish-filter-field" class="form-control js-filterable-field js-navigation-enable" placeholder="Filter branches/tags">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" data-filter-placeholder="Filter branches/tags" class="js-select-menu-tab" role="tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" data-filter-placeholder="Find a tag…" class="js-select-menu-tab" role="tab">Tags</a>
            </li>
          </ul>
        </div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches" role="menu">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open selected"
               href="/andrewt3000/MachineLearning/blob/master/neuralNets.md"
               data-name="master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg aria-hidden="true" class="octicon octicon-check select-menu-item-icon" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                master
              </span>
            </a>
        </div>

          <div class="select-menu-no-results">Nothing to show</div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div>

    </div>
  </div>
</div>

  <div class="BtnGroup float-right">
    <a href="/andrewt3000/MachineLearning/find/master"
          class="js-pjax-capture-input btn btn-sm BtnGroup-item"
          data-pjax
          data-hotkey="t">
      Find file
    </a>
    <button aria-label="Copy file path to clipboard" class="js-zeroclipboard btn btn-sm BtnGroup-item tooltipped tooltipped-s" data-copied-hint="Copied!" type="button">Copy path</button>
  </div>
  <div class="breadcrumb js-zeroclipboard-target">
    <span class="repo-root js-repo-root"><span class="js-path-segment"><a href="/andrewt3000/MachineLearning"><span>MachineLearning</span></a></span></span><span class="separator">/</span><strong class="final-path">neuralNets.md</strong>
  </div>
</div>


  <div class="commit-tease">
      <span class="float-right">
        <a class="commit-tease-sha" href="/andrewt3000/MachineLearning/commit/bc9b2767ef0d5c2da6b170da52b8539b1d3ea0c0" data-pjax>
          bc9b276
        </a>
        <relative-time datetime="2016-11-30T08:35:36Z">Nov 30, 2016</relative-time>
      </span>
      <div>
        <img alt="@andrewt3000" class="avatar" height="20" src="https://avatars2.githubusercontent.com/u/838741?v=3&amp;s=40" width="20" />
        <a href="/andrewt3000" class="user-mention" rel="author">andrewt3000</a>
          <a href="/andrewt3000/MachineLearning/commit/bc9b2767ef0d5c2da6b170da52b8539b1d3ea0c0" class="message" data-pjax="true" title="Update neuralNets.md">Update neuralNets.md</a>
      </div>

    <div class="commit-tease-contributors">
      <button type="button" class="btn-link muted-link contributors-toggle" data-facebox="#blob_contributors_box">
        <strong>1</strong>
         contributor
      </button>
      
    </div>

    <div id="blob_contributors_box" style="display:none">
      <h2 class="facebox-header" data-facebox-id="facebox-header">Users who have contributed to this file</h2>
      <ul class="facebox-user-list" data-facebox-id="facebox-description">
          <li class="facebox-user-list-item">
            <img alt="@andrewt3000" height="24" src="https://avatars0.githubusercontent.com/u/838741?v=3&amp;s=48" width="24" />
            <a href="/andrewt3000">andrewt3000</a>
          </li>
      </ul>
    </div>
  </div>


<div class="file">
  <div class="file-header">
  <div class="file-actions">

    <div class="BtnGroup">
      <a href="/andrewt3000/MachineLearning/raw/master/neuralNets.md" class="btn btn-sm BtnGroup-item" id="raw-url">Raw</a>
        <a href="/andrewt3000/MachineLearning/blame/master/neuralNets.md" class="btn btn-sm js-update-url-with-hash BtnGroup-item">Blame</a>
      <a href="/andrewt3000/MachineLearning/commits/master/neuralNets.md" class="btn btn-sm BtnGroup-item" rel="nofollow">History</a>
    </div>


        <button type="button" class="btn-octicon disabled tooltipped tooltipped-nw"
          aria-label="You must be signed in to make or propose changes">
          <svg aria-hidden="true" class="octicon octicon-pencil" height="16" version="1.1" viewBox="0 0 14 16" width="14"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"/></svg>
        </button>
        <button type="button" class="btn-octicon btn-octicon-danger disabled tooltipped tooltipped-nw"
          aria-label="You must be signed in to make or propose changes">
          <svg aria-hidden="true" class="octicon octicon-trashcan" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M11 2H9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1H2c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1v9c0 .55.45 1 1 1h7c.55 0 1-.45 1-1V5c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm-1 12H3V5h1v8h1V5h1v8h1V5h1v8h1V5h1v9zm1-10H2V3h9v1z"/></svg>
        </button>
  </div>

  <div class="file-info">
      124 lines (79 sloc)
      <span class="file-info-divider"></span>
    11.8 KB
  </div>
</div>

  
  <div id="readme" class="readme blob instapaper_body">
    <article class="markdown-body entry-content" itemprop="text"><h3><a id="user-content-neural-networks" class="anchor" href="#neural-networks" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Neural Networks</h3>

<p>Neural Network - an acyclical directed graph. Neural nets have input layers, hidden layers and output layers. Layers are connected by weighted synapsis that multiply their input times the weight. Hidden layer consists of neurons that sum their inputs from synapsis and execute an activation function on the sum. The weights are intially set to random values but are trained with backpropagation.  The input and output are of fixed size. They are often called artificial neural networks, to distinguish it from biological nerons. Also called feedforward neural network to distinguish from more complicated neural nets with feedback mechanisms such as recurrent neural networks. </p>

<p><a href="https://github.com/andrewt3000/MachineLearning/blob/master/img/nn.png" target="_blank"><img src="https://github.com/andrewt3000/MachineLearning/raw/master/img/nn.png" height="250px" width="250px" style="max-width:100%;"></a></p>

<p><a href="http://frnsys.com/ai_notes/machine_learning/neural_nets.html">neural nets</a> thorough and concise study notes about neural networks.   </p>

<p><a href="https://www.youtube.com/watch?v=bxe2T-V8XRs">Neural Networks demystified video</a> - videos explaining neural networks. Includes <a href="https://github.com/stephencwelch/Neural-Networks-Demystified">notes</a>.    </p>

<p><a href="http://playground.tensorflow.org/#activation=tanh&amp;batchSize=10&amp;dataset=circle&amp;regDataset=reg-plane&amp;learningRate=0.03&amp;regularizationRate=0&amp;noise=0&amp;networkShape=4,2&amp;seed=0.28720&amp;showTestData=false&amp;discretize=false&amp;percTrainData=50&amp;x=true&amp;y=true&amp;xTimesY=false&amp;xSquared=false&amp;ySquared=false&amp;cosX=false&amp;sinX=false&amp;cosY=false&amp;sinY=false&amp;collectStats=false&amp;problem=classification&amp;initZero=false">TensorFlow Neural Network Playground</a>  - This demo lets you run a neural network in your browser and see results graphically. Be sure to click the play button to start training. The network can easily train for the first three datasets with default parameters but the challenge is to get the network to train to the spiral dataset.  </p>

<p><a href="https://classroom.udacity.com/courses/ud730/">Deep learning on Udacity</a>  </p>

<h3><a id="user-content-features--input" class="anchor" href="#features--input" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Features / Input</h3>

<p>Features - measurable property being observed. In neural net context, it's  the input to a neural network.  Examples of features are pixel brightness in image object recognition, words encoded as vectors in nlp applications, audio signal in voice recognition applications.  </p>

<p>Feature selection - The process of choosing the features. It is important to pick features that correlate with the output. </p>

<p>Dimensionality reduction - Reducing number of variables.  A simple example is selecting the area of a house as a feature rather than using width and length seperately. Other examples include singular value decomposition, auto-encoders, and t-SNE (for visualizations).      </p>

<p>Feature scaling - scale each feature to be in a common range typically -1 to 1 where 0 is the mean value.    </p>

<h3><a id="user-content-hyperparameters" class="anchor" href="#hyperparameters" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Hyperparameters</h3>

<p>Hyperparameters - the model’s parameters in a neural net such as architecture, learning rate, and regularization factor.  </p>

<p>Architecture - The structure of a neural network i.e. number of hidden layers, and number of nodes. </p>

<p>Number of hidden layers - the higher the number of layers the more layers of abstraction it can represent. too many layers and the the network suffers from the vanishing or exploding gradient problem.  </p>

<p>Learning rate (α) - controls the size of the adjustments made during the training process. A typical value is .1 but often the value is a smaller number.<br>
if α is too low, convergance is slow.
if α is too high, there is no convergance, because it overshoots the local minimum.<br>
The learning rate is often reduced to a smaller number over time. This is often called annealing or decay. (examples: step decay, exponential decay)  </p>

<p>Underfitting - output doesn't fit the training data well.<br>
Overfitting - output fits training data well, but doesn't work well on test data.  </p>

<p>Regularization - a technique to minimize overfitting.  </p>

<p>L1 uses sum of absolute value of weights. L1 can yield sparse outputs.<br>
L2 uses sum of squared weights. L2 can't yield sparse outputs.    </p>

<p><a href="https://www.cs.toronto.edu/%7Ehinton/absps/JMLRdropout.pdf">Dropout</a> - a form of regularization. "The key idea is to randomly drop units (along with their connections) from the neural network during training." Typical hyperparameter value is .5 (50%). As dropout value approaches zero, dropout has less effect, as it approaches 1 there are more connections are being zeroed out. The remaining active connections are scaled up to compensate for the zeroed out connections. See <a href="https://iamtrask.github.io/2015/07/28/dropout/">Hinton's dropout in 3 lines of python</a> which features the following example:   </p>

<div class="highlight highlight-source-python"><pre>layer_1 <span class="pl-k">*=</span> np.random.binomial([np.ones((<span class="pl-c1">len</span>(<span class="pl-c1">X</span>),hidden_dim))],<span class="pl-c1">1</span><span class="pl-k">-</span>dropout_percent)[<span class="pl-c1">0</span>] <span class="pl-k">*</span> (<span class="pl-c1">1.0</span><span class="pl-k">/</span>(<span class="pl-c1">1</span><span class="pl-k">-</span>dropout_percent))</pre></div>

<h3><a id="user-content-activation-functions" class="anchor" href="#activation-functions" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Activation Functions</h3>

<p>Activation function - the "neuron" in the neural network executes an activation function on the sum of the weighted inputs. Typical activation functions include sigmoid, tanh, and ReLu.  </p>

<p>Sigmoid activation functions outputs a value between 0 and 1.  </p>

<div class="highlight highlight-source-python"><pre><span class="pl-c">#sigmoid activation function using numpy</span>
<span class="pl-k">def</span> <span class="pl-en">sigmoid</span>(<span class="pl-smi">z</span>):
    <span class="pl-k">return</span> <span class="pl-c1">1</span><span class="pl-k">/</span>(<span class="pl-c1">1</span><span class="pl-k">+</span>np.exp(<span class="pl-k">-</span>z))</pre></div>

<p>Tanh activation function outputs value between -1 and 1.  </p>

<p>ReLu activation is typically the activation function used in state of the art convolutioanal neural nets for image processing. ReLu stands for rectified linear unit. It returns 0 for negative values, and the same number for positive values. for x &lt; 0, y = 0. for x&gt;0, y = x.  </p>

<div class="highlight highlight-source-python"><pre><span class="pl-c">#relu activation function.</span>
<span class="pl-c">#numpy maximum - Compare two arrays and returns a new array containing the element-wise maxima</span>
hidden_layer <span class="pl-k">=</span> np.maximum(<span class="pl-c1">0</span>, np.dot(<span class="pl-c1">X</span>, <span class="pl-c1">W</span>) <span class="pl-k">+</span> b)
</pre></div>

<p>The softmax function is often used as the output's activation function and is useful for modeling probability distributions for multiclass classification where outputs are mutually exclusive (MNIST is an example). Output values are in range [0, 1]. The sum of outputs is 1. Use with cross entropy cost function.  </p>

<div class="highlight highlight-source-python"><pre><span class="pl-k">def</span> <span class="pl-en">softmax</span>(<span class="pl-smi">x</span>):
    e2x <span class="pl-k">=</span> np.exp(x) 
    <span class="pl-k">return</span> e2x <span class="pl-k">/</span> np.sum(e2x, <span class="pl-v">axis</span> <span class="pl-k">=</span> <span class="pl-c1">0</span>)</pre></div>

<h3><a id="user-content-training-a-network" class="anchor" href="#training-a-network" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Training a network</h3>

<p>Training data - Input and labeled output used as training examples. Data is typically split into training data, cross validation data and test data. Typical mix is 60% training, 20% validation and 20% testing data. Validation is used to tune the model and it's hyperparameters. Testing uses data that the model was never trained on.  </p>

<p>Training a network - minimize a cost function. Use backpropagation and gradient descent to adjust weights to make model more accurate. </p>

<p>Steps to training a network.  </p>

<ul>
<li>initialize weights and biases.<br></li>
<li>training data is entered feedforward.<br></li>
<li>error is backpropagated.</li>
<li>weights and biases are adjust based on learning rate.<br></li>
</ul>

<p>Number of times to iterate over the training data - You run the program until it hopefully converges on an acceptablely low error level. An epoch means the network has been been trained on every example once. You want to stop training if the validation data has an increasing error rate, this indicates overfitting. This is called early termination.   </p>

<p>Cost/error/objective/loss Function - measures how inaccurate a model is. Training a model minimizes the cost function. Sum of squared errors is a common cost function for regression. Cross entropy (aka log loss, negative log probability) is cost function for softmax function and multinomial logistic classification.   </p>

<p>Cross entropy function is sum of all the target values times the log of their output. Assuming the target output is 0 for all the wrong answers, and 1 for the correct answer, the correct answer is the only value that will contribute to the sum. The cost will be 0 if the correct output value is 1 because the log(1) is 0. The error approaches infinity as the output approaches 0 because the log of zero approaches infinity. See <a href="https://www.youtube.com/watch?v=mlaLLQofmR8">Geoffery Hinton lecture</a>  </p>

<p>Backpropagation - Apply the chain rule to compute the gradients (partial derivative) of the loss function with respect to the weights in the network by moving backwards (output to input) through the network.  </p>

<h4><a id="user-content-optimization-algorithms" class="anchor" href="#optimization-algorithms" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Optimization algorithms</h4>

<p>Batch gradient descent - Gradient descent is iteratively adjusting the weight by learning rate times the gradient to mimimize the error function. The term batch refers to the fact it uses the entire dataset. Batch works well for small datasets that have convex errors functions.  </p>

<p>Stochastic gradient descent is a variation of gradient descent that uses a single randomly choosen example to make an update to the weights. sgd is more scalable than batch graident descent and is used more often in practice for large scale deep learning. It's random nature makes it unlikely to get stuck in a local minima.  </p>

<p>Mini batch gradient descent: Stochastic gradient descent that considers more than one randomly choosen example befor making an update. Batch size is a hyperparmeter that determines how many training examples you consider before making a weight update. Typical values are factors of 2, such as 32 or 128.  </p>

<p>Momentum sgd is a variation that makes sgd less likely to go in the wrong direction because it collects data on each update in a velocity vector to assist in calculating the gradient. The velocity matrix represents the momentum. μ is a hyperparameter that represents the friction. μ is in the range of 0 to 1 and μ=1 is no friction.  </p>

<p>Other optimization algorithms include nesterov momentum sgd, adagrad, and adaDelta. See <a href="http://sebastianruder.com/optimizing-gradient-descent/">An overview of gradient descent optimization algorithms</a></p>

<p><a href="http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html">Practical tips for deep learning</a>  </p>

<h3><a id="user-content-other-types-of-neural-networks" class="anchor" href="#other-types-of-neural-networks" aria-hidden="true"><svg aria-hidden="true" class="octicon octicon-link" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>Other Types of Neural Networks</h3>

<p>Convolutional Neural Networks - Specialized to process a grid of information such as an image. Convolution neural networks use filters (aka kernels) that convolve over the grid.<br>
<a href="https://github.com/andrewt3000/MachineLearning/blob/master/cnn4Images.md">My notes on CNNs</a></p>

<p>Recurrent Neural Network (RNN) - Used for input sequences such as text, audio, or video. RNNs are similar to a vanilla neural network but they also pass the hidden state as output of each neuron via a weighted connection as an input to the neurons in the same layer during the next sequence. RNNs are trained by backpropagation through time.  This feedback architecture allows the network to have memory of previous inputs. The memory is limited by vanishing/exploding gradient problem. Exploding gradient can be resolved by gradient clipping. A common hyperparameter is the number of steps to go back or "unroll" during training. There are variations such as bi-directional and recursive RNNs. 
<a href="https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/">Code an RNN</a>  </p>

<p>LSTM - <a href="http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf">Long Short Term Memory</a> - A specialized RNN that is capable of long term dependencies and mitigates the vanishing gradient problem. It contains memory cells and gate units. The number of memory cells is a hyperparameter. Memory cells pass memory information forward. The gates decide what information is stored in the memory cells. A vanilla LSTM has a forget gates, input gates and output gates. There are <a href="http://arxiv.org/pdf/1503.04069.pdf">many variations of the LSTM</a>.<br>
<a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/">Understanding LSTM Networks</a> Blog post by Chris Olah.<br>
<a href="https://www.tensorflow.org/versions/r0.10/tutorials/recurrent/index.html">RNN/LSTM tutorial for TensorFlow</a>  </p>

<p>GRU - Gated Recurrent Unit - Introduced by Cho. Another RNN variant similar but simpler than LSTM. It contains one update gate and combines the hidden state and memory cells among other differences.  </p>

<p>Beam Search - keep track of several of the most probable sequences and then choose the highest probability path. This lowers the risk a single bad probability in the sequence getting you on the wrong track. <a href="https://classroom.udacity.com/courses/ud730/lessons/6378983156/concepts/63733319420923#">See udacity video</a>  </p>
</article>
  </div>

</div>

<button type="button" data-facebox="#jump-to-line" data-facebox-class="linejump" data-hotkey="l" class="d-none">Jump to Line</button>
<div id="jump-to-line" style="display:none">
  <!-- '"` --><!-- </textarea></xmp> --></option></form><form accept-charset="UTF-8" action="" class="js-jump-to-line-form" method="get"><div style="margin:0;padding:0;display:inline"><input name="utf8" type="hidden" value="&#x2713;" /></div>
    <input class="form-control linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
    <button type="submit" class="btn">Go</button>
</form></div>

  </div>
  <div class="modal-backdrop js-touch-events"></div>
</div>


    </div>
  </div>

    </div>

        <div class="container site-footer-container">
  <div class="site-footer" role="contentinfo">
    <ul class="site-footer-links float-right">
        <li><a href="https://github.com/contact" data-ga-click="Footer, go to contact, text:contact">Contact GitHub</a></li>
      <li><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
      <li><a href="https://shop.github.com" data-ga-click="Footer, go to shop, text:shop">Shop</a></li>
        <li><a href="https://github.com/blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a href="https://github.com/about" data-ga-click="Footer, go to about, text:about">About</a></li>

    </ul>

    <a href="https://github.com" aria-label="Homepage" class="site-footer-mark" title="GitHub">
      <svg aria-hidden="true" class="octicon octicon-mark-github" height="24" version="1.1" viewBox="0 0 16 16" width="24"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
    <ul class="site-footer-links">
      <li>&copy; 2016 <span title="0.04353s from github-fe118-cp1-prd.iad.github.net">GitHub</span>, Inc.</li>
        <li><a href="https://github.com/site/terms" data-ga-click="Footer, go to terms, text:terms">Terms</a></li>
        <li><a href="https://github.com/site/privacy" data-ga-click="Footer, go to privacy, text:privacy">Privacy</a></li>
        <li><a href="https://github.com/security" data-ga-click="Footer, go to security, text:security">Security</a></li>
        <li><a href="https://status.github.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a href="https://help.github.com" data-ga-click="Footer, go to help, text:help">Help</a></li>
    </ul>
  </div>
</div>



    

    <div id="ajax-error-message" class="ajax-error-message flash flash-error">
      <svg aria-hidden="true" class="octicon octicon-alert" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.865 1.52c-.18-.31-.51-.5-.87-.5s-.69.19-.87.5L.275 13.5c-.18.31-.18.69 0 1 .19.31.52.5.87.5h13.7c.36 0 .69-.19.86-.5.17-.31.18-.69.01-1L8.865 1.52zM8.995 13h-2v-2h2v2zm0-3h-2V6h2v4z"/></svg>
      <button type="button" class="flash-close js-flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
        <svg aria-hidden="true" class="octicon octicon-x" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
      </button>
      You can't perform that action at this time.
    </div>


      <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/compat-30ce4c86c27f88c3d1b4eb03efda59b45d8d7c871880dee0b8f73d5ef1b25fdf.js"></script>
      <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/frameworks-786af1c64a30401bf0a16eafed51415b9bcb03f72ff3183a1fa6007d7cb0f979.js"></script>
      <script async="async" crossorigin="anonymous" src="https://assets-cdn.github.com/assets/github-ad3fb51259f13f60fd093a3a2ab4f06ff39eb98af4ea390a85b31c598ad485ec.js"></script>
      
      
      
      
    <div class="js-stale-session-flash stale-session-flash flash flash-warn flash-banner d-none">
      <svg aria-hidden="true" class="octicon octicon-alert" height="16" version="1.1" viewBox="0 0 16 16" width="16"><path fill-rule="evenodd" d="M8.865 1.52c-.18-.31-.51-.5-.87-.5s-.69.19-.87.5L.275 13.5c-.18.31-.18.69 0 1 .19.31.52.5.87.5h13.7c.36 0 .69-.19.86-.5.17-.31.18-.69.01-1L8.865 1.52zM8.995 13h-2v-2h2v2zm0-3h-2V6h2v4z"/></svg>
      <span class="signed-in-tab-flash">You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
      <span class="signed-out-tab-flash">You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
    </div>
    <div class="facebox" id="facebox" style="display:none;">
  <div class="facebox-popup">
    <div class="facebox-content" role="dialog" aria-labelledby="facebox-header" aria-describedby="facebox-description">
    </div>
    <button type="button" class="facebox-close js-facebox-close" aria-label="Close modal">
      <svg aria-hidden="true" class="octicon octicon-x" height="16" version="1.1" viewBox="0 0 12 16" width="12"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48z"/></svg>
    </button>
  </div>
</div>

  </body>
</html>

