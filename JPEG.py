from PIL import Image
import numpy as np
import math
import os

def aumentaTamanho(img_YCBCR):
    (la, ca, p) = img_YCBCR.shape
    (ln, cn) = la,ca
    
    if la % 8 != 0:
        ln = la + (8 - (la % 8))
    
    if ca % 8 != 0:
        cn = ca + (8 - (ca % 8))
    
    if ln >= cn:
        cn = ln
    else:
        ln = cn
    
    img = np.zeros(shape = (ln,cn,p), dtype=int)
    img[:la, :ca, 0] = img_YCBCR[:,:,0]
    img[:la, :ca, 1] = img_YCBCR[:,:,1]
    img[:la, :ca, 2] = img_YCBCR[:,:,2]
    
    return img

def diminuiTamanho(img, l, c, p):
    final = np.zeros(shape=(l, c, p), dtype='uint8')
    final[:,:,:] = img[:l, :c, :]
    return final
    

def fazDCT(img):
    
    p, l, a, b = img.shape
    
    dct = np.zeros(shape=img.shape, dtype=float)
    img = img.astype(float)
    
    for z in range(p):
        for i in range(l):
            
            for u in range(8):
                for v in range(8):
            
                    # CALCULA O SOMATÓRIO
                    somatorio = 0
                    for x in range(8):
                        for y in range(8):
                            somatorio = somatorio + img[z,i,x,y]*math.cos(((2*x+1)*u*math.pi)/16)*math.cos(((2*y+1)*v*math.pi)/16)
                            
                    c_u = 1
                    if u == 0:
                        c_u = 1 / math.sqrt(2)
                    
                    c_v = 1
                    if v == 0:
                        c_v = 1 / math.sqrt(2)
                    
                    dct[z, i, u, v] = 0.25 * c_u * c_v * somatorio
              
    return np.round(dct).astype(int)

def fazIDCT(img):
    
    p, l, a, b = img.shape
    idct = np.zeros(shape=img.shape, dtype=float)
    img = img.astype(float)
    
    for z in range(p):
        for i in range(l):
            
            for x in range(8):
                for y in range(8):
                    
                    # CALCULA O SOMATÓRIO
                    somatorio = 0
                    for u in range(8):
                        for v in range(8):
                            
                            if u == 0:
                                c_u = 1 / math.sqrt(2)
                            else:
                                c_u = 1
        
                            if v == 0:
                                c_v = 1 / math.sqrt(2)
                            else:
                                c_v = 1
        
                            somatorio = somatorio + c_u*c_v*img[z,i,u,v]*math.cos((2*x+1)*u*math.pi/16)*math.cos((2*y+1)*v*math.pi/16)
                
                    idct[z,i,x,y] = 0.25 * somatorio
            
    return np.round(idct).astype(int)

def divide8x8(img):
    
    l, c, p = img.shape
    imgDiv = []
    
    for z in range(p):
        aux = []
        for i in range(l//8):
            for j in range(c//8):
                l_start = i*8
                l_end   = i*8 + 8
                c_start = j*8
                c_end   = j*8 + 8
                aux.append(img[l_start:l_end, c_start:c_end, z])
    
        imgDiv.append(aux)
       
    return np.array(imgDiv, dtype=int)

def junta8x8(img):
    
    p, l, a, b = img.shape
    
    imgJunta = np.zeros(shape=(int(math.sqrt(l)*8), int(math.sqrt(l)*8), p), dtype=int)
    
    for z in range(p):
        ii = 0
        jj = 0
        for i in range(l):
            
            l_start = ii*8
            l_end   = ii*8 + 8
            c_start = jj*8
            c_end   = jj*8 + 8
                
            imgJunta[l_start:l_end, c_start:c_end, z] = img[z, i, :, :]
            
            jj = jj + 1
            
            if jj == math.sqrt(l):
                ii = ii + 1
                jj = 0
        
    return imgJunta

def gerarMatrizQuantizacao(fatorQuantizacao):
    Q = np.zeros(shape=(8,8), dtype=int)
    for i in range(8):
        for j in range(8):
            Q[i,j] = 1 + (1 + i + j) * fatorQuantizacao

    return Q

def gerarDCTQuantized(img, Q):
    
    p, l, a, b = img.shape
    
    dctQuantized = np.zeros(shape=img.shape, dtype=float)
    img = img.astype(float)
    
    for z in range(p):
        for i in range (l):
            dctQuantized[z, i] = img[z, i] / Q
    
    return np.round(dctQuantized).astype(int)

def gerarDCTDesquantized(img, Q):
    
    p, l, a, b = img.shape
    
    dctDesquantized = np.zeros(shape=img.shape, dtype=int)
    
    for z in range(p):
        for i in range (l):
            dctDesquantized[z, i] = img[z, i] * Q
    
    return dctDesquantized

def RGBtoYCBCR(img):
    
    p, l, a, b = img.shape
    
    imgYCBCR = np.zeros(shape=img.shape, dtype=float)
    
    img = img.astype(float)
    
    for i in range(l):
        
        r = img[0, i]/255 
        g = img[1, i]/255
        b = img[2, i]/255
        
        y = 0.299*r + 0.587*g + 0.114*b
        cb = -0.168736*r - 0.331264*g + 0.5*b
        cr = 0.5*r - 0.418688*g - 0.081312*b
        #cb = preprocessing.minmax_scale(cb, feature_range=(-0.5, 0.5))
        #cr = preprocessing.minmax_scale(cr, feature_range=(-0.5, 0.5))
        
        Y =  (y*219) + 16
        CB = (cb*224) + 128 
        CR = (cr*224) + 128
        
        imgYCBCR[0, i] = Y
        imgYCBCR[1, i] = CB
        imgYCBCR[2, i] = CR

    return np.round(imgYCBCR).astype(int)

def YCBCRtoRGB(img):
    p, l, a, b = img.shape
    
    imgRGB = np.zeros(shape=img.shape, dtype=float)
    img = img.astype(float)
    
    for i in range(l):
        
        y = (img[0, i]-16)/219 
        cb = (img[1, i]-128)/224
        cr = (img[2, i]-128)/224
        
        r = y + 1.402*cr
        g = y - 0.344136*cb - 0.714136*cr
        b = y + 1.772*cb
        
        R = r*255
        G = g*255
        B = b*255
        
        imgRGB[0, i] = R
        imgRGB[1, i] = G
        imgRGB[2, i] = B

    return np.round(imgRGB).astype(int)

def codificaZigzag(img):
    
    p, l, a, b = img.shape
    
    img_zigzag = np.zeros(shape=(p, l, 64), dtype=int)
    
    for z in range(p):
        for w in range(l):
            i = 0
            j = 0
            cont = 0
            while True:
                
                cont = sobeDiagonal(img, img_zigzag, cont, z, w, i, j, True)
                
                i, j = j, i
                
                if j == 7 and i == 7:
                    break
                
                
                if j == 7:
                    i += 1
                else:
                    j += 1
                
                cont = desceDiagonal(img, img_zigzag, cont, z, w, i, j, True)
                
                i, j = j, i
                
                if  i == 7:
                    j += 1
                else:
                    i += 1
    
    return img_zigzag

def desceDiagonal(img, img_zigzag, cont, z, w, i, j, flag):
    
    y = j
    for x in range(i, j+1):
        if flag:
            img_zigzag[z, w, cont] = img[z, w, x, y]
        else:
            img_zigzag[z, w, x, y] = img[z, w, cont]
        cont += 1
        y -= 1
        
    return cont

def sobeDiagonal(img, img_zigzag, cont, z, w, i, j, codifica):
    
    x = i
    for y in range(j, i+1):
        if codifica:
            img_zigzag[z, w, cont] = img[z, w, x, y]
        else:
            img_zigzag[z, w, x, y] = img[z, w, cont]
        cont += 1
        x -= 1
    return cont

def decodificaZigzag(img):
    
    p, l, a = img.shape
    
    img_zigzag = np.zeros(shape=(p, l, 8, 8), dtype=int)
    
    for z in range(p):
        for w in range(l):
            i = 0
            j = 0
            cont = 0
            while True:
                
                cont = sobeDiagonal(img, img_zigzag, cont, z, w, i, j, False)
                
                i, j = j, i
                
                if j == 7 and i == 7:
                    break
                
                
                if j == 7:
                    i += 1
                else:
                    j += 1
                
                cont = desceDiagonal(img, img_zigzag, cont, z, w, i, j, False)
                
                i, j = j, i
                
                if  i == 7:
                    j += 1
                else:
                    i += 1
    
    return img_zigzag

def codificaRLE(img):
    
    (p, l, a) = img.shape
    
    matrizRLE = []
    
    for z in range(p):
        for i in range(l):
            
            j = 0
            
            aux = list()
            while j < 64:    
                cont = 0
                pos = j
                while pos < 64 and img[z, i, j] == img[z, i, pos]:
                    pos = pos + 1
                    cont = cont + 1
                
                x = (cont, img[z, i, j])
                aux.append(x)
                j = pos
            
            matrizRLE.append(aux)   
            
    return matrizRLE

def decodificaRLE(img):
    
    matrizRLE = np.zeros(shape=(3, int(len(img)/3), 64), dtype=int)
    
    z = 0
    i = 0
    for listas in img:
        
        cont = 0
        for tuplas in listas:
            rep, value = tuplas
            
            if value != 0:
                for pp in range(rep):
                    matrizRLE[z, i%( int(len(img)/3)), cont] = value
                    cont += 1
            else:
                cont += rep
        i += 1
        if i%( int(len(img)/3)) == 0:
            z += 1
    
    return matrizRLE

def main():
     
    # NOME SE REFERE AO NOME DA IMAGEM QUE SERÁ COMPACTADA
    nome = 'naruto'
    # COMPRESSAO É O VALOR USADO PARA GERAR A MATRIZ DE QUANTIZAÇÃO
    # QUANTO MAIOR MAIS PERDA TERÁ A IMAGEM
    compressao = 20
    
    img = Image.open(nome+'.jpg')
    
    print('Input file size   : ', img.size )
    print('Input file name   : ', nome+'.jpg' )
    print('Input Image Size  : ', os.path.getsize (nome+'.jpg'), 'bytes')
    
    img = np.array(img, dtype=np.uint8)[:,:,:3]
    
    (l, c, p) = img.shape
    
    img = aumentaTamanho(img)
    
    img = divide8x8(img)
    
    img = RGBtoYCBCR(img)
    
    img = img - 128
    
    img = fazDCT(img)
    
    Q = gerarMatrizQuantizacao(compressao)
    
    img = gerarDCTQuantized(img, Q)
    
    img = codificaZigzag(img)
    
    img = codificaRLE(img)
    
    img = decodificaRLE(img)
    
    img = decodificaZigzag(img) 
    
    img = gerarDCTDesquantized(img, Q)
    
    img = fazIDCT(img)
    
    img = img + 128
    
    img = YCBCRtoRGB(img)
    img[img>255] = 255
    img[img<0] = 0
    
    img = junta8x8(img)
    
    img = diminuiTamanho(img, l, c, p)
    
    im = Image.fromarray(img)
    print('-------------------------------------------------------------------')
    im.save(nome+'compress.jpg')
    print('Input file size   : ', im.size )
    print('Input file name   : ', nome+'compress.jpg' )
    print('Input Image Size  : ', os.path.getsize (nome+'compress.jpg'), 'bytes')
    
    return
    
if __name__ == '__main__': # chamada da funcao principal
    main() # chamada da função main