from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.text import * 

#export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://drive.google.com/uc?export=download&id=1B8IZXkHKPSl9xfr_xracE4K3Ee0SYSE9'
export_file_name = 'model'
export_file_url2 = 'https://drive.google.com/uc?export=download&id=1676PCFeIcw6N7xwZa-CdmJokFJGlIbv5'
export_file_name2 = 'lm'

#classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    await download_file(export_file_url2, path/export_file_name2)
    try:
        #data_lm2 = load_data(path, export_file_name2)
        #learn = language_model_learner(data_lm2, AWD_LSTM, drop_mult=0.5)
        #learn.load(export_file_name)
        return 0
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

def predict(learn, text:str, n_words:int=10,  no_unk:bool=True, min_p:float=None, sep:str=' ',decoder=decode_spec_tokens):
  ds = learn.data.single_dl.dataset
  learn.model.reset()
  xb,yb = learn.data.one_item(text)
  new_idx = []
  res = learn.pred_batch(batch=(xb,yb))[0][-1]
  if no_unk: res[learn.data.vocab.stoi[UNK]] = 0.
  if min_p is not None: 
      if (res >= min_p).float().sum() == 0:
          warn(f"There is no item with probability >= {min_p}, try a lower value.")
      else: res[res < min_p] = 0.
  top5ans  = list(np.asarray(torch.topk(res,n_words,0)[1]))
  top5prob = list(np.asarray(torch.topk(res,n_words,0)[0]))
  res = decoder(learn.data.vocab.textify(top5ans, sep=None))
  res = " , ".join(res)
  return  res


#@app.route('/')
#def index(request):
#    html = path/'view'/'index.html'
#    return HTMLResponse(html.open().read())


@app.route('/')
async def analyze(request):
    data = await request.form()
    #img_bytes = await (data['file'].read())
    #img = open_image(BytesIO(img_bytes))
    prediction = predict(learn,data)
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
