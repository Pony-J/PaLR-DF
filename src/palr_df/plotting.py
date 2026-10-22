"""Regenerate the seven paper result figures from immutable archived records.

Display names follow the manuscript; archived model identifiers remain intact.
The layout comes from the original manuscript plotting scripts. Times New Roman
is used when installed; DejaVu Serif is the portable fallback.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import font_manager
from .io import write_json

def figures(repo, output):
    frozen=Path(repo)/'results/frozen'
    OUT=Path(output)
    OUT.mkdir(parents=True,exist_ok=True)
    FIGURES=[]
    available={f.name for f in font_manager.fontManager.ttflist}
    font='Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({'font.family':font,'font.size':9,'axes.labelsize':9.5,'axes.titlesize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'axes.unicode_minus':True,'mathtext.fontset':'dejavuserif','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65,'savefig.facecolor':'white'})
    BLUE='#25689A';ORANGE='#C56F2C';GRAY='#707B85';GREEN='#26806A';RED='#B94F4F';BLACK='#202020';W=3.261111111111111
    NAMES={'SSA-TCN Approx.-D':'SSA-TCN-D','Informer Approx.-D':'Informer','Informer-D':'Informer','LSTM-Attention-D':'LSTM-ATT-D','MGAMNet Approx.-D':'MGAMNet'}
    spec=pd.read_csv(frozen/'spectra.csv')
    spec=spec[spec.model=='V2']
    sweep=pd.read_csv(frozen/'rank_reference_3seed.csv').rename(columns={'macro_rmse_ns':'macro_rmse_ns_mean','sample_sd_ns':'macro_rmse_ns_sample_std'})
    confirm=pd.read_csv(frozen/'rank_reference_5seed.csv').rename(columns={'macro_rmse_ns':'macro_rmse_ns_mean','sample_sd_ns':'macro_rmse_ns_sample_std'})
    delta=pd.read_csv(frozen/'satellite_differences.csv')
    daily=pd.read_csv(frozen/'c41_daily.csv')
    assert len(spec)==280 and len(delta)==28
    def style(ax,grid='y'):
     ax.spines[['top','right']].set_visible(False);ax.spines[['bottom','left']].set_color(GRAY);ax.set_axisbelow(True);ax.grid(axis=grid,color='#E1E4E7',lw=.45);ax.tick_params(length=2,pad=2)
    def plotted_data(fig):
     def vals(a):return np.ma.filled(np.ma.asarray(a),np.nan).tolist()
     rows=[]
     for ax in fig.axes:
      rows.append({'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
       'xticks':vals(ax.get_xticks()),'yticks':vals(ax.get_yticks()),
       'lines':[{'x':vals(x.get_xdata()),'y':vals(x.get_ydata())} for x in ax.lines],
       'collections':[{'type':type(x).__name__,'offsets':vals(x.get_offsets()),
        'segments':[vals(v) for v in x.get_segments()] if hasattr(x,'get_segments') else []} for x in ax.collections],
       'bars':[[p.get_x(),p.get_y(),p.get_width(),p.get_height()] for p in ax.patches if hasattr(p,'get_height')]})
     return rows

    def adjust_layout(fig,name):
     if name=='01_ordinary_rank':
      fig.set_size_inches(W,1.95)
      fig.subplots_adjust(left=.155,right=.99,bottom=.27,top=.72,wspace=.23)
     elif name=='02_weighted_ranks':
      fig.set_size_inches(W,2.32)
      fig.subplots_adjust(left=.14,right=.985,bottom=.29,top=.65,wspace=.20)
      for ax in fig.axes:ax.title.set_position((.5,1.35))
      for t in fig.texts:
       if t.get_text().startswith('Nominal'):t.set_position((.5,.99));t.set_va('top')
       elif t.get_text().startswith('Above'):t.set_position((.5,.91));t.set_va('top')
       elif t.get_text().startswith('28 satellites'):t.set_position((.5,.135))
      fig.legends[0].set_bbox_to_anchor((.53,.005))
     elif name=='03_rank_sweep':
      fig.set_size_inches(W,2.15)
      fig.subplots_adjust(left=.165,right=.99,bottom=.24,top=.70,wspace=.50)
      fig.legends[0].set_bbox_to_anchor((.53,1.03))
      fig.legends[1].set_bbox_to_anchor((.53,.925))
     elif name=='04_paired_comparison':
      fig.set_size_inches(W,2.32)
      ax=fig.axes[0];ax.set_position([.31,.23,.36,.61])
      for t in ax.texts:
       if 'Win/Tie' in t.get_text() or 'Raw' in t.get_text():
        t.set_position((t.get_position()[0],1.08));t.set_va('center')
     elif name=='05_accuracy_complexity':
      fig.set_size_inches(W,2.60)
      fig.subplots_adjust(left=.18,right=.98,bottom=.19,top=.94)
      offsets={'r=2':(28000,.653),'r=3':(45000,.515),'r=6':(44500,.685),
       'r=12':(67000,.620),'V2':(110000,.636),'INT-Best':(195000,.507),
       'LSTM-ATT-D':(65000,.895),'LSTM-D':(220000,.875),'CNN-D':(780000,.820),
       'Informer':(260000,.753),'MGAMNet':(600000,.687),'SSA-TCN-D':(730000,.571)}
      for t in fig.axes[0].texts:
       label='r=3' if t.get_text()=='PaLR-DF\n(r=3)' else t.get_text()
       if label in offsets:t.set_position(offsets[label])
     elif name=='07_c41_daily':
      pass

    def save(fig,name,old_media):
     before=plotted_data(fig)
     adjust_layout(fig,name)
     after=plotted_data(fig)
     assert before==after, 'Layout changed plotted values: '+name
     fig.canvas.draw();ren=fig.canvas.get_renderer();texts=[];outside=[]
     for t in fig.findobj(matplotlib.text.Text):
      if not t.get_visible() or not t.get_text().strip():continue
      bb=t.get_window_extent(ren)
      if bb.x1<0 or bb.y1<0:continue
      texts.append({'text':t.get_text(),'font_pt':t.get_fontsize(),'font_family':t.get_fontfamily(),'bbox_px':list(bb.bounds)})
      if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:outside.append(t.get_text())
     for ext in ['png','pdf','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=300)
     FIGURES.append({'name':name,'replaces_media':old_media,'width_in':fig.get_figwidth(),
      'height_in':fig.get_figheight(),'min_visible_font_pt':min(t['font_pt'] for t in texts),
      'outside_canvas_text':outside,'text_objects':texts,'plotted_data_unchanged':before==after,
      'plotted_data':after})
     plt.close(fig)

    # 1. Keep checkpoint counts and the original 0--80 vertical scale.
    fig,axes=plt.subplots(1,2,figsize=(W,2.05),sharey=True);fig.subplots_adjust(left=.15,right=.99,bottom=.28,top=.74,wspace=.2)
    for i,(h,ax) in enumerate(zip(['12h','24h'],axes)):
     vals=spec[spec.horizon==h].raw_r95.to_numpy();counts=np.bincount(vals.astype(int),minlength=6)
     assert len(vals)==140
     ax.bar(range(6),counts,color=BLUE,width=.72,zorder=3)
     for j,n in enumerate(counts):ax.text(j,n+1.5,str(n),ha='center',fontsize=8.5)
     ax.set_xticks(range(6));ax.set_ylim(0,83);ax.set_yticks([0,20,40,60,80]);ax.set_title(f'({chr(97+i)}) {h[:-1]} h',pad=21)
     ax.text(.5,1.02,f'Median {np.median(vals):g}; max 5',transform=ax.transAxes,ha='center',fontsize=8.5)
     style(ax)
    axes[0].set_ylabel('Checkpoint count');fig.supxlabel('95% energy effective rank',y=.07,fontsize=9.5)
    save(fig,'01_ordinary_rank','image2.png')
    # 2. Horizontal jitter only; all 840 checkpoint/method values retained.
    fig,axes=plt.subplots(1,2,figsize=(W,2.65),sharey=True);fig.subplots_adjust(left=.14,right=.98,bottom=.25,top=.66,wspace=.18)
    rng=np.random.default_rng(20260916);points=[]
    for i,(h,ax) in enumerate(zip(['12h','24h'],axes)):
     part=spec[spec.horizon==h].sort_values(['satellite','seed'])
     for k,(col,color) in enumerate([('raw_r95',GRAY),('context_r95',ORANGE),('task_r95',BLUE)]):
      v=part[col].to_numpy();xx=np.zeros(len(v))
      for rank in np.unique(v):
       inds=np.flatnonzero(v==rank);n=len(inds);off=-.29+.52*(np.arange(n)+.5+rng.uniform(-.18,.18,n))/n;rng.shuffle(off);xx[inds]=k+off
      ax.scatter(xx,v,s=2.2,c=color,alpha=.72,linewidth=0,zorder=3);med=np.median(v);p90=np.quantile(v,.9)
      ax.hlines(med,k+.29,k+.415,color=BLACK,lw=1.2);ax.scatter([k+.505],[p90],marker='D',s=10,facecolors='white',edgecolors=BLACK,lw=.7)
      ax.text(k+.065,1.04,f'{med:g}\n{p90:g}',transform=ax.get_xaxis_transform(),ha='center',fontsize=8.5,linespacing=1.25)
      points.extend(dict(horizon=h,method=col,satellite=s,seed=int(sd),rank=float(y),plot_x=float(x)) for s,sd,y,x in zip(part.satellite,part.seed,v,xx))
     ax.set_xlim(-.44,2.64);ax.set_ylim(-.3,5.5);ax.set_xticks(np.arange(3)+.065,['Ord.','Context','Task']);ax.tick_params(axis='x',length=0,labelsize=8.5);ax.set_yticks(range(6));ax.set_title(f'({chr(97+i)}) {h[:-1]} h',y=1.38,pad=0);style(ax)
    axes[0].set_ylabel('95% energy rank');fig.text(.5,.963,'Nominal factorization rank = 48',ha='center',fontsize=9)
    fig.text(.5,.895,'Above axes: median / P90',ha='center',fontsize=8.5)
    fig.text(.5,.065,'28 satellites × 5 seeds per horizon',ha='center',fontsize=8.5)
    fig.legend(handles=[Line2D([],[],color=BLACK,lw=1.2,label='Median'),Line2D([],[],color=BLACK,marker='D',mfc='white',lw=0,label='P90')],loc='lower center',bbox_to_anchor=(.53,-.025),ncol=2,frameon=False,handlelength=1.1)
    save(fig,'02_weighted_ranks','image5.png')
    # 3. Rank sweep, preserving the original vertical limits.
    fig,axes=plt.subplots(1,2,figsize=(W,2.5));fig.subplots_adjust(left=.16,right=.98,bottom=.27,top=.65,wspace=.48)
    for i,(h,ax) in enumerate(zip(['12h','24h'],axes)):
     s=sweep[sweep.horizon==h].sort_values('rank');f=confirm[confirm.horizon==h].sort_values('rank')
     ax.errorbar(s['rank'],s.macro_rmse_ns_mean,yerr=s.macro_rmse_ns_sample_std,fmt='o-',color=GRAY,mfc='white',ms=3,lw=.8,capsize=2)
     for _,r in f.iterrows():ax.errorbar(r['rank'],r.macro_rmse_ns_mean,yerr=r.macro_rmse_ns_sample_std,fmt='o',color=ORANGE if r['rank']==3 else BLUE,ms=3.8,lw=.9,capsize=2)
     ax.set_xticks([2,3,4,6,8,12]);ax.set_xlim(1.2,12.8);ax.set_ylim((.54,.609) if h=='12h' else (.96,1.14));ax.set_yticks([.54,.56,.58,.60] if h=='12h' else [.96,1,1.04,1.08,1.12]);ax.set_title(f'({chr(97+i)}) {h[:-1]} h',pad=5);style(ax)
    axes[0].set_ylabel('Macro RMSE (ns)');fig.supxlabel(r'Residual-shape rank $r$',y=.065,fontsize=9.5)
    handles=[Line2D([],[],marker='o',color=GRAY,mfc='white',lw=.8,label='3 seeds ± SD'),Line2D([],[],marker='o',color=BLUE,lw=0,label='5 seeds ± SD')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,1.01),ncol=2,frameon=False,handlelength=1,columnspacing=.9)
    fig.legend(handles=[Line2D([],[],marker='o',color=ORANGE,lw=0,label='PaLR-DF (r=3)')],loc='upper center',bbox_to_anchor=(.53,.91),frameon=False,handlelength=1)
    save(fig,'03_rank_sweep','image6.png')
    # 4. Archived paired statistics; no reselection or recomputation of tests.
    pairs=pd.read_csv(frozen/'paired_12h_with_MGAMNet.csv')
    order=['SSA-TCN Approx.-D','MGAMNet Approx.-D','Informer Approx.-D','CNN-D','LSTM-D','LSTM-Attention-D']
    pairs=pairs.set_index('reference').loc[order].reset_index()
    fig=plt.figure(figsize=(W,2.65));ax=fig.add_axes([.315,.23,.35,.58]);ax.set_ylim(-.6,5.6);ax.set_xlim(-.43,.02)
    for i,r in pairs.iterrows():
     y=5-i;ax.errorbar(r.paired_mean_difference_ns,y,xerr=[[r.paired_mean_difference_ns-r.bootstrap_ci95_low_ns],[r.bootstrap_ci95_high_ns-r.paired_mean_difference_ns]],fmt='o',color=BLUE,ms=3,lw=.9,capsize=2)
     ax.text(1.24,y,f'{r.wins}/{r.ties}/{r.losses}',transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=8.5)
     pv='<0.0001' if r.wilcoxon_two_sided_p<.0001 else f'{r.wilcoxon_two_sided_p:.4f}';ax.text(1.69,y,pv,transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=8.5)
    ax.set_yticks(range(6),[NAMES.get(m,m) for m in order[::-1]]);ax.set_xticks([-.4,-.2,0]);ax.axvline(0,color=GRAY,ls='--',lw=.7);style(ax,grid='x');ax.grid(False,axis='y')
    ax.text(1.24,1.1,'Win/Tie/\nLoss',transform=ax.transAxes,ha='center',fontsize=8.5);ax.text(1.69,1.1,'Raw\np-value',transform=ax.transAxes,ha='center',fontsize=8.5)
    fig.text(.5,.10,'Paired mean difference (ns)',ha='center',fontsize=9.5);fig.text(.5,.035,'PaLR-DF minus baseline; 95% CI',ha='center',fontsize=9)
    save(fig,'04_paired_comparison','image7.png')
    # 5. All external models and the historical internal references retained.
    fig,ax=plt.subplots(figsize=(W,3.15));fig.subplots_adjust(left=.18,right=.98,bottom=.19,top=.94)
    points=[(NAMES.get(r.model,r.model),r.parameters,r.macro_rmse_ns,r.color) for r in pd.read_csv(frozen/'complexity_points.csv').itertuples()]

    offsets={'r=2':(29500,.653),'r=3':(43000,.51),'r=6':(44500,.685),'r=12':(59500,.615),'V2':(100000,.627),'INT-Best':(175000,.511),'LSTM-ATT-D':(60000,.897),'LSTM-D':(200000,.863),'CNN-D':(530000,.805),'Informer':(260000,.753),'MGAMNet':(520000,.687),'SSA-TCN-D':(730000,.587)}
    for name,x,y,c in points:
     ax.scatter([x],[y],s=15,c=c,zorder=4);label='PaLR-DF\n(r=3)' if name=='r=3' else name
     ax.annotate(label,(x,y),xytext=offsets[name],ha='center',va='center',fontsize=8.5,color=BLACK,arrowprops={'arrowstyle':'-','color':c,'lw':.65},annotation_clip=False)
    ax.set_xscale('log');ax.set_xlim(22000,1550000);ax.set_ylim(.485,.927);ax.set_xticks([3e4,1e5,3e5,1e6],['30k','100k','300k','1M']);ax.minorticks_off();ax.set_yticks([.5,.6,.7,.8,.9]);ax.set_xlabel('Parameters per model (log scale)');ax.set_ylabel('12 h Macro RMSE (ns)');style(ax)
    save(fig,'05_accuracy_complexity','image8.png')
    # 6. Current manuscript Fig. 9: single INT-Best panel.
    fig,ax=plt.subplots(figsize=(W,1.78))
    fig.subplots_adjust(left=.17,right=.995,bottom=.35,top=.72)
    vals=delta.LR3_minus_DTD.to_numpy();xx=np.arange(len(delta))
    ax.bar(xx,vals,color=[GREEN if x<0 else RED for x in vals],width=.74,zorder=3)
    ax.axhline(0,color=BLACK,lw=.6)
    ax.set_xticks(xx,delta.satellite,rotation=90,fontsize=8.5)
    ax.set(xlim=(-.8,27.8),ylim=(-.3,1.1),yticks=[-.2,0,.5,1],ylabel='ΔRMSE (ns)')
    ax.set_title('PaLR-DF vs INT-Best',fontsize=9.5,loc='left',pad=3)
    style(ax)
    fig.legend(handles=[Patch(facecolor=GREEN,label='PaLR-DF lower'),Patch(facecolor=RED,label='PaLR-DF higher')],loc='upper center',bbox_to_anchor=(.56,1.02),frameon=False,ncol=2,handlelength=.8,columnspacing=.7)
    save(fig,'06_satellite_differences','image9.png')
    # 7. Original C41 historical diagnostic; no new C41 evaluation.
    fig,ax=plt.subplots(figsize=(W,2.5));fig.subplots_adjust(left=.16,right=.99,bottom=.25,top=.82);xx=np.arange(4)
    for dx,col,c,name in [(-.18,'LR3',BLUE,'PaLR-DF'),(.18,'V7R',ORANGE,'INT-Best')]:
     bars=ax.bar(xx+dx,daily[col],width=.32,color=c,label=name,zorder=3)
     for b,v in zip(bars,daily[col]):ax.text(b.get_x()+b.get_width()/2+(-.045 if dx<0 else .045),v+.12,f'{v:.3f}',ha='center',fontsize=8.5)
    ax.set_xticks(xx,[f'Jan {27+i}\n{int(n)} windows' for i,n in enumerate(daily.windows_per_seed)]);ax.set_ylim(0,7.4);ax.set_yticks([0,2,4,6]);ax.set_ylabel('Mean window RMSE (ns)');ax.set_xlabel('Forecast target start date in 2024');style(ax);fig.legend(loc='upper center',bbox_to_anchor=(.56,1.01),ncol=2,frameon=False)
    save(fig,'07_c41_daily','image10.png')

    numbers=[2,5,6,7,8,9,10]
    for number, record in zip(numbers,FIGURES):
        record['paper_figure']=number
        record['png_sha256']=hashlib.sha256((OUT/(record['name']+'.png')).read_bytes()).hexdigest()
        record.pop('replaces_media',None)
    report={'figure_count':len(FIGURES),'formats':['png','pdf','svg'],'font':font,'scope':'historical first-paper figures; no new evaluation','figures':FIGURES}
    report=json.loads(json.dumps(report,default=lambda x:x.item(),allow_nan=False))
    write_json(OUT/'figure_manifest.json',report)
    return {'figure_count':len(FIGURES),'formats':report['formats'],'paper_figures':numbers}
