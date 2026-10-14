// Shared CUDA/Metal math. GPU address spaces and atomics are supplied by callers.
DEVICE bool gut_ray(int x,int y,CPTR k,TPTR r) {
  float a=(x+.5f-k[2])/k[0],b=(y+.5f-k[3])/k[1];
  if(k[8]<.5f){float n=SQRT(a*a+b*b+1);r[0]=a/n;r[1]=b/n;r[2]=1/n;return true;}
  float rd=SQRT(a*a+b*b),t=FMIN(rd,1.5706963f);
  for(int j=0;j<20;j++){float q=t*t,v=t*(1+q*(k[4]+q*(k[5]+q*(k[6]+q*k[7]))));
    float d=1+q*(3*k[4]+q*(5*k[5]+q*(7*k[6]+q*9*k[7])));
    t=FMIN(1.5706963f,FMAX(0.f,t-(v-rd)/FMAX(d,1.e-8f)));}
  float q=t*t,v=t*(1+q*(k[4]+q*(k[5]+q*(k[6]+q*k[7]))));
  if(ABS(v-rd)>1.e-5f||t>=1.5705963f)return false;
  float s=rd>1.e-12f?SIN(t)/rd:1.f;r[0]=a*s;r[1]=b*s;r[2]=COS(t);return true;
}
DEVICE void gut_project_point(TCPTR p,CPTR k,TPTR uv){
  float rr=SQRT(p[0]*p[0]+p[1]*p[1]),s;
  if(k[8]<.5f)s=1.f/FMAX(p[2],1.e-8f);
  else {float t=ATAN2(rr,p[2]),q=t*t;s=rr>1.e-12f?t*(1+q*(k[4]+q*(k[5]+q*(k[6]+q*k[7]))))/rr:1.f/FMAX(p[2],1.e-8f);}
  uv[0]=k[0]*p[0]*s+k[2];uv[1]=k[1]*p[1]*s+k[3];
}
DEVICE void gut_project(int i,int h,int w,CPTR means,CPTR cov,CPTR k,WPTR bounds){
  float m[3]={means[3*i],means[3*i+1],means[3*i+2]};
  if(m[2]<=k[9]){bounds[4*i]=1;bounds[4*i+1]=1;bounds[4*i+2]=0;bounds[4*i+3]=0;return;}
  float l[9]={0,0,0,0,0,0,0,0,0};int o=9*i;
  l[0]=SQRT(FMAX(cov[o],1.e-16f));l[3]=cov[o+3]/l[0];l[6]=cov[o+6]/l[0];
  l[4]=SQRT(FMAX(cov[o+4]-l[3]*l[3],1.e-16f));l[7]=(cov[o+7]-l[6]*l[3])/l[4];
  l[8]=SQRT(FMAX(cov[o+8]-l[6]*l[6]-l[7]*l[7],1.e-16f));
  // Conservative fallback for near-plane-crossing/large particles.
  float maxsd=SQRT(FMAX(cov[o],FMAX(cov[o+4],cov[o+8])));
  if(m[2]-3*maxsd<=k[9]||maxsd/m[2]>.2f){bounds[4*i]=0;bounds[4*i+1]=0;bounds[4*i+2]=w-1;bounds[4*i+3]=h-1;return;}
  float u[6],v[6],mu=0,mv=0;
  // UT: n=3, alpha=1, kappa=0, lambda=0; center weight is zero.
  for(int a=0;a<3;a++)for(int sign=0;sign<2;sign++){
    float p[3],uv[2],s=sign?1.7320508f:-1.7320508f;
    for(int j=0;j<3;j++)p[j]=m[j]+s*l[3*j+a];
    gut_project_point(p,k,uv);int b=2*a+sign;u[b]=uv[0];v[b]=uv[1];mu+=uv[0]/6;mv+=uv[1]/6;
  }
  float cu=0,cv=0;for(int a=0;a<6;a++){cu+=(u[a]-mu)*(u[a]-mu)/6;cv+=(v[a]-mv)*(v[a]-mv)/6;}
  float ru=3*SQRT(cu)+4,rv=3*SQRT(cv)+4;
  bounds[4*i]=FMAX(0.f,FLOOR(mu-ru));bounds[4*i+1]=FMAX(0.f,FLOOR(mv-rv));
  bounds[4*i+2]=FMIN(float(w-1),CEIL(mu+ru));bounds[4*i+3]=FMIN(float(h-1),CEIL(mv+rv));
}
DEVICE float gut_response(int i,TCPTR ray,CPTR m,CPTR precision,CPTR opa,TPTR d,TPTR pd){
  float pr[3],pm[3],den=0,num=0;
  for(int j=0;j<3;j++){pr[j]=0;pm[j]=0;for(int a=0;a<3;a++){pr[j]+=precision[9*i+3*j+a]*ray[a];pm[j]+=precision[9*i+3*j+a]*m[3*i+a];}den+=ray[j]*pr[j];num+=ray[j]*pm[j];}
  float t=num/FMAX(den,1.e-12f);if(t<=0)return 0;
  for(int j=0;j<3;j++)d[j]=t*ray[j]-m[3*i+j];
  float rho=0;for(int j=0;j<3;j++){pd[j]=0;for(int a=0;a<3;a++)pd[j]+=precision[9*i+3*j+a]*d[a];rho+=d[j]*pd[j];}
  return rho<9?FMIN(.999f,FMAX(0.f,opa[i]*EXP(-.5f*FMAX(rho,0.f)))):0;
}
DEVICE void gut_forward(int p,int h,int w,CPTR m,CPTR prec,CPTR color,CPTR opa,CPTR k,IPTR starts,IPTR ids,WPTR out){
  int x=p%w,y=p/w,tile=(y/16)*((w+15)/16)+x/16;float ray[3],rgb[3]={0,0,0},trans=1;
  if(gut_ray(x,y,k,ray))for(int j=starts[tile];j<starts[tile+1];j++){
    int i=ids[j];float d[3],pd[3];float a=gut_response(i,ray,m,prec,opa,d,pd);
    for(int c=0;c<3;c++)rgb[c]+=trans*a*color[3*i+c];trans*=1-a;
  }
  for(int c=0;c<3;c++)out[4*p+c]=rgb[c];out[4*p+3]=1-trans;
}
DEVICE void gut_backward(int p,int h,int w,CPTR m,CPTR prec,CPTR color,CPTR opa,CPTR k,IPTR starts,IPTR ids,CPTR image,CPTR grad,APTR gm,APTR gp,APTR gc,APTR go){
  int x=p%w,y=p/w,tile=(y/16)*((w+15)/16)+x/16;float ray[3];if(!gut_ray(x,y,k,ray))return;
  float prefix[3]={0,0,0},trans=1,total_trans=1-image[4*p+3];
  for(int j=starts[tile];j<starts[tile+1];j++){
    int i=ids[j];float d[3],pd[3];float a=gut_response(i,ray,m,prec,opa,d,pd);if(a<=0)continue;
    float da=grad[4*p+3]*total_trans/(1-a);
    for(int c=0;c<3;c++){
      prefix[c]+=trans*a*color[3*i+c];float remaining=image[4*p+c]-prefix[c];
      da+=grad[4*p+c]*(trans*color[3*i+c]-remaining/(1-a));
      ADD(gc+3*i+c,grad[4*p+c]*trans*a);
    }
    if(a<.999f&&opa[i]>0){
      ADD(go+i,da*a/opa[i]);
      for(int c=0;c<3;c++){ADD(gm+3*i+c,da*a*pd[c]);for(int b=0;b<3;b++)ADD(gp+9*i+3*c+b,-.5f*da*a*d[c]*d[b]);}
    }
    trans*=1-a;
  }
}
