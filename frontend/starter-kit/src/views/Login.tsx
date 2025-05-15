'use client'

// React Imports
import { useState } from 'react'

// Next Imports
import { useRouter } from 'next/navigation'

// MUI Imports
import useMediaQuery from '@mui/material/useMediaQuery'
import { styled, useTheme } from '@mui/material/styles'
import Typography from '@mui/material/Typography'
import IconButton from '@mui/material/IconButton'
import InputAdornment from '@mui/material/InputAdornment'
import Checkbox from '@mui/material/Checkbox'
import Button from '@mui/material/Button'
import FormControlLabel from '@mui/material/FormControlLabel'
import Divider from '@mui/material/Divider'

// Third-party Imports
import classnames from 'classnames'

// Type Imports
import type { SystemMode } from '@core/types'

// Component Imports
import Link from '@components/Link'
import Logo from '@components/layout/shared/Logo'
import CustomTextField from '@core/components/mui/TextField'

// Config Imports
import themeConfig from '@configs/themeConfig'

// Hook Imports
import { useImageVariant } from '@core/hooks/useImageVariant'
import { useSettings } from '@core/hooks/useSettings'

// Styled Custom Components
const LoginIllustration = styled('img')(({ theme }) => ({
  zIndex: 2,
  blockSize: 'auto',
  maxBlockSize: 680,
  maxInlineSize: '100%',
  margin: theme.spacing(12),
  [theme.breakpoints.down(1536)]: {
    maxBlockSize: 550
  },
  [theme.breakpoints.down('lg')]: {
    maxBlockSize: 450
  }
}))

const MaskImg = styled('img')({
  blockSize: 'auto',
  maxBlockSize: 355,
  inlineSize: '100%',
  position: 'absolute',
  insetBlockEnd: 0,
  zIndex: -1
})

const LoginV2 = ({ mode }: { mode: SystemMode }) => {
  // States
  const [isPasswordShown, setIsPasswordShown] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // Vars
  const darkImg = '/images/pages/auth-mask-dark.png'
  const lightImg = '/images/pages/auth-mask-light.png'
  const darkIllustration = '/images/illustrations/auth/v2-login-dark.png'
  const lightIllustration = '/images/illustrations/auth/v2-login-light.png'
  const borderedDarkIllustration = '/images/illustrations/auth/v2-login-dark-border.png'
  const borderedLightIllustration = '/images/illustrations/auth/v2-login-light-border.png'

  // Hooks
  const router = useRouter()
  const { settings } = useSettings()
  const theme = useTheme()
  const hidden = useMediaQuery(theme.breakpoints.down('md'))
  const authBackground = useImageVariant(mode, lightImg, darkImg)

  const characterIllustration = useImageVariant(
    mode,
    lightIllustration,
    darkIllustration,
    borderedLightIllustration,
    borderedDarkIllustration
  )

  const handleClickShowPassword = () => setIsPasswordShown(show => !show)

  // 我添加的
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const res = await fetch('http://127.0.0.1:5000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })

      if (res.ok) {
        const data = await res.json()
        localStorage.setItem('token', data.token)
        router.push('/home')
      } else if (res.status === 401) {
        setError('用户名或密码错误')
      } else {
        setError('登录失败，请稍后重试')
      }
    } catch {
      setError('网络错误，请稍后再试')
    } finally {
      setLoading(false)
    }
  }



//   return (
//     <div className='flex bs-full justify-center'>
//       <div
//         className={classnames(
//           'flex bs-full items-center justify-center flex-1 min-bs-[100dvh] relative p-6 max-md:hidden',
//           {
//             'border-ie': settings.skin === 'bordered'
//           }
//         )}
//       >
//         <LoginIllustration src={characterIllustration} alt='character-illustration' />
//         {!hidden && (
//           <MaskImg
//             alt='mask'
//             src={authBackground}
//             className={classnames({ 'scale-x-[-1]': theme.direction === 'rtl' })}
//           />
//         )}
//       </div>
//       <div className='flex justify-center items-center bs-full bg-backgroundPaper !min-is-full p-6 md:!min-is-[unset] md:p-12 md:is-[480px]'>
//         <Link className='absolute block-start-5 sm:block-start-[33px] inline-start-6 sm:inline-start-[38px]'>
//           <Logo />
//         </Link>
//         <div className='flex flex-col gap-6 is-full sm:is-auto md:is-full sm:max-is-[400px] md:max-is-[unset] mbs-11 sm:mbs-14 md:mbs-0'>
//           <div className='flex flex-col gap-1'>
//             <Typography variant='h4'>{`Welcome to ${themeConfig.templateName}! 👋🏻`}</Typography>
//             <Typography>Please sign-in to your account and start the adventure</Typography>
//           </div>
//           <form
//             noValidate
//             autoComplete='off'
//             onSubmit={e => {
//               e.preventDefault()
//               router.push('/home')
//             }}
//             className='flex flex-col gap-5'
//           >
//             <CustomTextField autoFocus fullWidth label='Email or Username' placeholder='Enter your email or username' />
//             <CustomTextField
//               fullWidth
//               label='Password'
//               placeholder='············'
//               id='outlined-adornment-password'
//               type={isPasswordShown ? 'text' : 'password'}
//               InputProps={{
//                 endAdornment: (
//                   <InputAdornment position='end'>
//                     <IconButton edge='end' onClick={handleClickShowPassword} onMouseDown={e => e.preventDefault()}>
//                       <i className={isPasswordShown ? 'tabler-eye-off' : 'tabler-eye'} />
//                     </IconButton>
//                   </InputAdornment>
//                 )
//               }}
//             />
//             <div className='flex justify-between items-center gap-x-3 gap-y-1 flex-wrap'>
//               <FormControlLabel control={<Checkbox />} label='Remember me' />
//               <Typography className='text-end' color='primary' component={Link}>
//                 Forgot password?
//               </Typography>
//             </div>
//             <Button fullWidth variant='contained' type='submit'>
//               Login
//             </Button>
//             <div className='flex justify-center items-center flex-wrap gap-2'>
//               <Typography>New on our platform?</Typography>
//               <Typography component={Link} color='primary'>
//                 Create an account
//               </Typography>
//             </div>
//             <Divider className='gap-2 text-textPrimary'>or</Divider>
//             <div className='flex justify-center items-center gap-1.5'>
//               <IconButton className='text-facebook' size='small'>
//                 <i className='tabler-brand-facebook-filled' />
//               </IconButton>
//               <IconButton className='text-twitter' size='small'>
//                 <i className='tabler-brand-twitter-filled' />
//               </IconButton>
//               <IconButton className='text-textPrimary' size='small'>
//                 <i className='tabler-brand-github-filled' />
//               </IconButton>
//               <IconButton className='text-error' size='small'>
//                 <i className='tabler-brand-google-filled' />
//               </IconButton>
//             </div>
//           </form>
//         </div>
//       </div>
//     </div>
//   )
// }




return (
    <div className='flex bs-full justify-center'>
      <div
        className={classnames(
          'flex bs-full items-center justify-center flex-1 min-bs-[100dvh] relative p-6 max-md:hidden',
          {
            'border-ie': settings.skin === 'bordered',
          }
        )}
      >
        <LoginIllustration src={characterIllustration} alt='character-illustration' />
        {!hidden && (
          <MaskImg
            alt='mask'
            src={authBackground}
            className={classnames({ 'scale-x-[-1]': theme.direction === 'rtl' })}
          />
        )}
      </div>
      <div className='flex justify-center items-center bs-full bg-backgroundPaper !min-is-full p-6 md:!min-is-[unset] md:p-12 md:is-[480px]'>
        <Link className='absolute block-start-5 sm:block-start-[33px] inline-start-6 sm:inline-start-[38px]'>
          <Logo />
        </Link>
        <div className='flex flex-col gap-6 is-full sm:is-auto md:is-full sm:max-is-[400px] md:max-is-[unset] mbs-11 sm:mbs-14 md:mbs-0'>
          <div className='flex flex-col gap-1'>
            <Typography variant='h4'>{`Welcome to ${themeConfig.templateName}! 👋🏻`}</Typography>
            <Typography>Please sign-in to your account and start the adventure</Typography>
          </div>
          <form noValidate autoComplete='off' onSubmit={handleSubmit} className='flex flex-col gap-5'>
            <CustomTextField
              autoFocus
              fullWidth
              label='Email or Username'
              placeholder='Enter your email or username'
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
            />
            <CustomTextField
              fullWidth
              label='Password'
              placeholder='············'
              id='outlined-adornment-password'
              type={isPasswordShown ? 'text' : 'password'}
              value={password}
              onChange={e => setPassword(e.target.value)}
              InputProps={{
                endAdornment: (
                  <InputAdornment position='end'>
                    <IconButton edge='end' onClick={handleClickShowPassword} onMouseDown={e => e.preventDefault()}>
                      <i className={isPasswordShown ? 'tabler-eye-off' : 'tabler-eye'} />
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              required
            />
            {error && (
              <Typography color='error' mt={1}>
                {error}
              </Typography>
            )}
            <div className='flex justify-between items-center gap-x-3 gap-y-1 flex-wrap'>
              <FormControlLabel control={<Checkbox />} label='Remember me' />
              <Typography className='text-end' color='primary' component={Link}>
                Forgot password?
              </Typography>
            </div>
            <Button fullWidth variant='contained' type='submit' disabled={loading}>
              {loading ? '登录中...' : 'Login'}
            </Button>
            <div className='flex justify-center items-center flex-wrap gap-2'>
              <Typography>New on our platform?</Typography>
              <Typography component={Link} color='primary'>
                Create an account
              </Typography>
            </div>
            <Divider className='gap-2 text-textPrimary'>or</Divider>
            <div className='flex justify-center items-center gap-1.5'>
              <IconButton className='text-facebook' size='small'>
                <i className='tabler-brand-facebook-filled' />
              </IconButton>
              <IconButton className='text-twitter' size='small'>
                <i className='tabler-brand-twitter-filled' />
              </IconButton>
              <IconButton className='text-textPrimary' size='small'>
                <i className='tabler-brand-github-filled' />
              </IconButton>
              <IconButton className='text-error' size='small'>
                <i className='tabler-brand-google-filled' />
              </IconButton>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}

export default LoginV2
