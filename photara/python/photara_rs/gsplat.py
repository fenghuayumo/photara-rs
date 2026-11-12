"""3D Gaussian cloud validation, lifecycle and standard PLY/.splat loading."""
from pathlib import Path
import numpy as np

C0 = 0.28209479177387814

def validate_gaussians(positions, scales, rotations=None, colors=None, opacities=None, sh_rest=None):
    p = np.ascontiguousarray(positions, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError('positions must have nonempty shape (N,3)')
    n = len(p)
    def array(value, shape, default):
        a = np.broadcast_to(default if value is None else value, shape).copy().astype(np.float32)
        if not np.isfinite(a).all(): raise ValueError('Gaussian attributes must be finite')
        return np.ascontiguousarray(a)
    if not np.isfinite(p).all(): raise ValueError('Nonfinite positions')
    s = array(scales, (n,3), 0.01)
    q = array(rotations, (n,4), (1,0,0,0))
    c = array(colors, (n,3), (0.5,0.5,0.5))
    opacity = array(opacities, (n,), 1)
    sh = np.zeros((n,15,3),np.float32)
    if sh_rest is not None:
        data = np.asarray(sh_rest,dtype=np.float32)
        if data.ndim != 3 or data.shape[0] != n or data.shape[2] != 3 or data.shape[1] not in (0,3,8,15):
            raise ValueError('sh_rest must have shape (N,0/3/8/15,3)')
        if not np.isfinite(data).all(): raise ValueError('Nonfinite spherical harmonics')
        sh[:,:data.shape[1]] = data
    norm = np.linalg.norm(q,axis=1)
    if np.any(norm <= 0) or not np.isfinite(norm).all(): raise ValueError('Invalid quaternion')
    if np.any(s < 0) or np.any(opacity < 0) or np.any(opacity > 1): raise ValueError('Invalid scale or opacity')
    return p,s,np.ascontiguousarray(q/norm[:,None]),np.column_stack((c,opacity)).astype(np.float32),sh.reshape(n,45)

class GaussianSplats:
    def __init__(self,scene,handle): self.scene,self.handle,self.removed=scene,handle,False
    def _session(self):
        self.scene._ensure_open()
        if self.removed: raise ValueError('Gaussian cloud was removed')
        return self.scene._session
    def set_transform(self,matrix): self._session().set_object_transform(self.handle,matrix)
    def set_visible(self,visible): self._session().set_object_visible(self.handle,visible)
    def remove(self):
        self._session().remove_object(self.handle)
        self.removed=True

def load_gaussians(path):
    path=Path(path)
    if path.suffix.lower()=='.splat':
        dtype=np.dtype([('p','<f4',3),('s','<f4',3),('c','u1',4),('q','u1',4)])
        if path.stat().st_size % 32: raise ValueError('Truncated .splat record')
        a=np.fromfile(path,dtype=dtype)
        data=dict(positions=a['p'],scales=a['s'],rotations=(a['q'].astype(np.float32)-128)/128,
                  colors=a['c'][:,:3]/255,opacities=a['c'][:,3]/255)
    elif path.suffix.lower()=='.ply':
        data=_load_ply(path)
    else: raise ValueError('Expected a .ply or .splat file')
    validate_gaussians(**data)
    return data

def _load_ply(path):
    types={'float':'f4','float32':'f4','double':'f8','float64':'f8','uchar':'u1','uint8':'u1',
           'char':'i1','int8':'i1','short':'i2','int16':'i2','ushort':'u2','uint16':'u2',
           'int':'i4','int32':'i4','uint':'u4','uint32':'u4'}
    with path.open('rb') as f:
        if f.readline().strip()!=b'ply': raise ValueError('Invalid PLY signature')
        props=[];count=None;fmt=None;vertex=False
        for _ in range(10000):
            line=f.readline()
            if not line: raise ValueError('Truncated PLY header')
            words=line.decode('ascii').split()
            if not words: continue
            if words[0]=='end_header': break
            if words[0]=='format': fmt=words[1]
            if words[0]=='element':
                if count is None and words[1]!='vertex' and int(words[2]): raise ValueError('PLY vertex element must be first')
                vertex=words[1]=='vertex'
                if vertex: count=int(words[2])
            if words[0]=='property' and vertex:
                if words[1] not in types: raise ValueError('Unsupported PLY vertex property')
                props.append((words[2],types[words[1]]))
        else: raise ValueError('Excessive PLY header')
        if count is None or count<=0: raise ValueError('Empty PLY cloud')
        names=[x[0] for x in props]
        if fmt=='ascii':
            rows=[f.readline().decode('ascii').split() for _ in range(count)]
            if any(len(row)!=len(props) for row in rows): raise ValueError('Truncated PLY vertices')
            a=np.asarray(rows,dtype=np.float32)
            fields={name:a[:,i] for i,name in enumerate(names)}
        elif fmt in ('binary_little_endian','binary_big_endian'):
            endian='<' if fmt=='binary_little_endian' else '>'
            a=np.fromfile(f,dtype=np.dtype([(name,endian+kind) for name,kind in props]),count=count)
            if len(a)!=count: raise ValueError('Truncated PLY vertices')
            fields={name:a[name].astype(np.float32) for name in names}
        else: raise ValueError('Unsupported PLY format')
    def columns(prefix,n):
        try:return np.column_stack([fields[prefix+str(i)] for i in range(n)])
        except KeyError as exc:raise ValueError(f'Missing 3DGS PLY property: {exc}') from None
    try:positions=np.column_stack([fields[k] for k in ('x','y','z')]);opacity=fields['opacity']
    except KeyError as exc:raise ValueError(f'Missing 3DGS PLY property: {exc}') from None
    scales=np.exp(columns('scale_',3).astype(np.float64))
    colors=.5+C0*columns('f_dc_',3)
    rest=sorted((int(k[7:]),k) for k in fields if k.startswith('f_rest_'))
    sh=None
    if rest:
        if [x[0] for x in rest]!=list(range(len(rest))) or len(rest) not in (9,24,45):raise ValueError('Unsupported SH coefficient layout')
        sh=columns('f_rest_',len(rest)).reshape(count,3,-1).transpose(0,2,1)
    # Stable sigmoid avoids overflowing trained opacity logits.
    alpha=np.exp(-np.logaddexp(0,-opacity.astype(np.float64)))
    return dict(positions=positions,scales=scales,rotations=columns('rot_',4),colors=colors,opacities=alpha,sh_rest=sh)


def save_gaussian_ply(path, **data):
    """Write a binary standard 3DGS PLY for interoperability with Blender.

    Zero scales/opacity are bounded to tiny positive values for finite logits.
    """
    p,s,q,c,sh=validate_gaussians(**data)
    n=len(p)
    names=['x','y','z']+[f'f_dc_{i}' for i in range(3)]+['opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]+[f'f_rest_{i}' for i in range(45)]
    alpha=np.clip(c[:,3].astype(np.float64),1e-12,1-1e-12)
    rows=np.column_stack((p,(c[:,:3]-.5)/C0,np.log(alpha)-np.log1p(-alpha),
                          np.log(np.maximum(s,1e-30)),q,sh.reshape(n,15,3).transpose(0,2,1).reshape(n,45)))
    with Path(path).open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(n)+'\n'+''.join('property float '+name+'\n' for name in names)+'end_header\n').encode('ascii'))
        f.write(rows.astype('<f4').tobytes())
